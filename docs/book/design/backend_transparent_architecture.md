---
title: "Backend-Transparent Architecture"
date: 2025-11-11
author: "Jukka Aho"
status: "Design Proposal"
last_updated: 2025-11-11
tags: ["design", "architecture", "GPU", "CPU", "backend", "transparency"]
---

## Design Principle: Backend Transparency

**Core principle:** Users should **never** see or care whether computation happens on CPU or GPU.

**What users write:**

```julia
using JuliaFEM

# Create problem
physics = Physics(Elasticity, "beam", 3)
add_elements!(physics, elements)
add_dirichlet!(physics, nodes, components, values)

# Solve - backend chosen automatically!
solution = solve!(physics)
```

**What happens behind the scenes:**

- If `using CUDA` + GPU available → GPU backend (50-100× faster)
- Otherwise → CPU backend with multithreading
- **User code identical** - no `solve_gpu!()` vs `solve_cpu!()`
- **No GPU types in user API** - no `GPUElasticityData`, `CuArray`, etc.

## Current Problems

### 1. GPU in Type Names

```julia
# ❌ BAD: Exposes implementation
struct GPUElasticityData
    nodes::CuArray{Float64,2}
    # ...
end
```

**Problem:** Name reveals backend. What about CPU? `CPUElasticityData`? Then user has two different types!

### 2. GPU in Function Names

```julia
# ❌ BAD: User must choose backend
solve_elasticity_gpu!(physics)
solve_elasticity_cpu!(physics)
```

**Problem:** User code must change based on hardware. Not portable!

### 3. GPU Types in Public API

```julia
# ❌ BAD: User sees CUDA types
function foo(data::GPUElasticityData)
    nodes = data.nodes  # CuArray!
end
```

**Problem:** Users must understand GPU programming. Violates abstraction!

## Proposed Solution: Three-Layer Architecture

### Layer 1: User API (Backend-Agnostic)

**User-facing types and functions - NO backend details:**

```julia
# Problem type (same for all backends)
physics = Physics(Elasticity, "problem_name", dimension)

# Add data (CPU arrays only - backend converts internally)
add_elements!(physics, elements::Vector{Element})
add_dirichlet!(physics, nodes::Vector{Int}, components::Vector{Int}, value)
add_neumann!(physics, surface_element::Element, traction::Vec)

# Solve (backend chosen automatically)
solution = solve!(physics; backend=Auto())  # or GPU(), CPU(4)
```

**Key:** All input/output uses **CPU arrays**. Backend conversion is internal.

### Layer 2: Backend Abstraction

**Abstract types defining interface:**

```julia
# Backend selection
abstract type AbstractBackend end
struct Auto <: AbstractBackend end      # Choose automatically
struct GPU <: AbstractBackend end       # Force GPU (error if unavailable)
struct CPU <: AbstractBackend           # Force CPU
    nthreads::Int
end

# Internal data (not exposed to user)
abstract type AbstractElasticityData end

# Implementations (in package extensions)
struct ElasticityDataGPU <: AbstractElasticityData
    nodes::CuArray{Float64,2}
    # ... GPU-resident data
end

struct ElasticityDataCPU <: AbstractElasticityData
    nodes::Matrix{Float64}
    # ... CPU data with multithreading
end
```

**Key:** Abstract interface, concrete implementations in extensions.

### Layer 3: Backend Implementations

**GPU backend (in `ext/JuliaFEMCUDAExt/`):**

```julia
module JuliaFEMCUDAExt

using JuliaFEM
using CUDA

# Only loaded if user has 'using CUDA'
struct ElasticityDataGPU <: AbstractElasticityData
    # ... CuArray fields
end

function initialize_backend(::GPU, physics::Physics{Elasticity})
    # Convert CPU data to GPU
    return ElasticityDataGPU(...)
end

function solve_backend!(data::ElasticityDataGPU, physics)
    # GPU kernels + CG solver
end

end  # module
```

**CPU backend (in `ext/JuliaFEMThreadsExt/` or built-in):**

```julia
struct ElasticityDataCPU <: AbstractElasticityData
    nodes::Matrix{Float64}
    elements::Matrix{Int32}
    # ... CPU arrays
end

function initialize_backend(::CPU, physics::Physics{Elasticity})
    # Keep data on CPU
    return ElasticityDataCPU(...)
end

function solve_backend!(data::ElasticityDataCPU, physics)
    # Threaded assembly + CG solver
end
```

## Implementation: User API

### Core solve function

```julia
"""
    solve!(physics::Physics{Elasticity}; backend=Auto(), kwargs...)

Solve elasticity problem. Backend is chosen automatically unless specified.

# Arguments
- `physics`: Problem definition with elements and BCs
- `backend`: Backend selection (Auto(), GPU(), or CPU(nthreads))
- `kwargs...`: Solver options (tol, max_iter, etc.)

# Returns
- `solution`: Solution struct with displacement field

# Examples

```julia
# Automatic backend selection (GPU if available, else CPU)
sol = solve!(physics)

# Force GPU backend (errors if GPU unavailable)
sol = solve!(physics; backend=GPU())

# Force CPU with 8 threads
sol = solve!(physics; backend=CPU(8))
```
"""
function solve!(physics::Physics{Elasticity};
                backend::AbstractBackend=Auto(),
                tol=1e-6,
                max_iter=1000)
    
    # 1. Choose backend
    backend_impl = select_backend(backend)
    
    # 2. Initialize backend-specific data
    data = initialize_backend(backend_impl, physics)
    
    # 3. Solve using backend
    u = solve_backend!(data, physics; tol, max_iter)
    
    # 4. Return solution (CPU array)
    return ElasticitySolution(physics, Array(u))  # Convert back to CPU
end
```

### Backend selection

```julia
function select_backend(::Auto)
    # Try GPU first
    if @isdefined(CUDA) && CUDA.functional()
        @info "Using GPU backend"
        return GPU()
    else
        nthreads = Threads.nthreads()
        @info "Using CPU backend with $nthreads threads"
        return CPU(nthreads)
    end
end

select_backend(::GPU) = begin
    if !(@isdefined(CUDA) && CUDA.functional())
        error("GPU backend requested but CUDA not available")
    end
    return GPU()
end

select_backend(cpu::CPU) = cpu
```

## Package Extension Structure

**Directory layout:**

```
JuliaFEM.jl/
├── src/
│   ├── JuliaFEM.jl           # Main module
│   ├── physics/
│   │   └── elasticity.jl     # Physics{Elasticity}, add_elements!, etc.
│   ├── backend/
│   │   ├── abstract.jl       # AbstractBackend, AbstractElasticityData
│   │   ├── selection.jl      # select_backend(), solve!()
│   │   └── cpu.jl            # CPU backend (always available)
│   └── ...
├── ext/
│   └── JuliaFEMCUDAExt/      # GPU backend (loaded only if CUDA available)
│       ├── JuliaFEMCUDAExt.jl
│       ├── elasticity_gpu.jl
│       └── kernels.jl
└── Project.toml              # Weak dependency on CUDA
```

**Project.toml (weak dependencies):**

```toml
[deps]
LinearAlgebra = "..."
Tensors = "..."
# ... other always-required deps

[weakdeps]
CUDA = "..."  # Only loaded if user does 'using CUDA'

[extensions]
JuliaFEMCUDAExt = "CUDA"
```

## Benefits

### 1. User Code Portability

**Same code runs on laptop or HPC cluster:**

```julia
# Works everywhere!
sol = solve!(physics)
```

No `if CUDA.functional()` checks, no platform-specific branches.

### 2. Performance Transparency

**Backend selection is automatic:**

```julia
# Laptop (no GPU):        → CPU backend (8 threads)
# Workstation (RTX 4090): → GPU backend (50× faster)
# HPC (no CUDA):          → CPU backend (64 threads)
```

User sees performance improvement without code changes!

### 3. Clean API

**No GPU types in user API:**

```julia
# ✅ GOOD: User only sees CPU arrays
nodes = [1.0 0.0 0.0; 0.0 1.0 0.0]  # Matrix{Float64}
elements = [Element(...), ...]       # Vector{Element}

physics = Physics(Elasticity, "prob", 3)
add_elements!(physics, elements)
sol = solve!(physics)

u = sol.u  # Vector{Float64} (always CPU!)
```

### 4. Gradual GPU Adoption

**Users can try GPU without changing code:**

```julia
# Day 1: CPU only
using JuliaFEM
sol = solve!(physics)  # Uses CPU

# Day 2: Install CUDA, get GPU speedup!
using JuliaFEM, CUDA
sol = solve!(physics)  # Automatically uses GPU!
```

No code rewrite required!

## Implementation Phases

### Phase 1: Refactor Current GPU Code ✅ (This PR)

**Rename types:**

- `GPUElasticityData` → `ElasticityDataGPU` (internal only)
- `solve_elasticity_gpu!()` → `solve_backend!(::ElasticityDataGPU, ...)`

**Add CPU backend:**

- Create `ElasticityDataCPU` with same interface
- Implement threaded assembly (Threads.@threads)
- Implement CPU CG solver

**Unified solve:**

- `solve!(physics; backend=Auto())` dispatches to correct backend

### Phase 2: Package Extensions (Week 2)

**Move GPU code to extension:**

- Create `ext/JuliaFEMCUDAExt/`
- Move CUDA-specific code (kernels, CuArray handling)
- Test: `using JuliaFEM` works without CUDA installed

**Benefits:**

- Smaller package size (no CUDA dependency unless needed)
- Faster loading time
- Cleaner dependency tree

### Phase 3: Backend Optimization (Weeks 3-4)

**Optimize CPU backend:**

- Threaded assembly (element loop parallelization)
- SIMD vectorization (basis function evaluation)
- Cache blocking (better memory access patterns)

**Optimize GPU backend:**

- Kernel fusion (reduce launches)
- Shared memory (reduce global memory traffic)
- Preconditioning (Chebyshev-Jacobi)

### Phase 4: Advanced Backends (Months 2-3)

**Multi-GPU:**

```julia
sol = solve!(physics; backend=MultiGPU([0, 1]))  # 2 GPUs
```

**Distributed CPU:**

```julia
sol = solve!(physics; backend=MPI(64))  # 64 MPI ranks
```

**Hybrid:**

```julia
sol = solve!(physics; backend=Hybrid(4, 2))  # 4 nodes × 2 GPUs/node
```

## Example: User Code (Before vs After)

### Before (Current - Backend Exposed) ❌

```julia
using JuliaFEM, CUDA

# User must know they're using GPU!
include("src/gpu_physics_elasticity.jl")
using .GPUElasticityPhysics

physics = Physics(Elasticity, "beam", 3)
add_elements!(physics, elements)
add_dirichlet!(physics, nodes, comps, 0.0)

# GPU-specific function!
sol = solve_elasticity_gpu!(physics)  # ← Hardcoded backend!

# What if no GPU? User must write:
if CUDA.functional()
    sol = solve_elasticity_gpu!(physics)
else
    sol = solve_elasticity_cpu!(physics)  # Different function!
end
```

**Problems:**

- User must understand GPU programming
- Code not portable (GPU-specific function)
- Manual backend selection (error-prone)

### After (Proposed - Backend Transparent) ✅

```julia
using JuliaFEM
# Optional: using CUDA  (enables GPU backend automatically)

physics = Physics(Elasticity, "beam", 3)
add_elements!(physics, elements)
add_dirichlet!(physics, nodes, comps, 0.0)

# One function, automatic backend!
sol = solve!(physics)  # ← Chooses GPU if available, else CPU

# Advanced: explicit backend control
sol = solve!(physics; backend=GPU())      # Force GPU
sol = solve!(physics; backend=CPU(8))     # Force CPU, 8 threads
sol = solve!(physics; backend=Auto())     # Automatic (default)
```

**Benefits:**

- User never sees "GPU" in their code
- Same code runs on any hardware
- Automatic optimal backend selection

## Type Names: Proposed Changes

### Before (❌ Exposes Backend)

```julia
struct GPUElasticityData      # ← "GPU" in public API!
    nodes::CuArray{...}       # ← CUDA type visible!
    # ...
end

solve_elasticity_gpu!(...)    # ← Backend in function name!
```

### After (✅ Backend Hidden)

**Internal types (not exported):**

```julia
# In ext/JuliaFEMCUDAExt/ (not visible to users)
struct ElasticityDataGPU <: AbstractElasticityData
    nodes::CuArray{...}
    # ...
end

# In src/backend/cpu.jl (not visible to users)
struct ElasticityDataCPU <: AbstractElasticityData
    nodes::Matrix{...}
    # ...
end
```

**User-facing API:**

```julia
# User only sees these (no backend details)
Physics{Elasticity}
solve!(physics; backend=Auto())
ElasticitySolution
```

## Naming Convention

**Internal backend types:**

- `ElasticityDataGPU` - GPU-resident data (CuArrays)
- `ElasticityDataCPU` - CPU-resident data (Matrix)
- `ElasticityDataMPI` - Distributed data (future)

**Suffix indicates implementation, not user-facing interface!**

**User types (no backend suffix):**

- `Physics{Elasticity}` - Problem definition (backend-agnostic)
- `ElasticitySolution` - Solution (always CPU arrays)
- `Material`, `BoundaryCondition`, etc. - All backend-agnostic

## Related Design Decisions

### 1. Data Transfer

**Automatic GPU transfer:**

```julia
# User provides CPU arrays
elements = [Element(...), ...]  # Vector{Element} on CPU

# solve! converts internally
sol = solve!(physics; backend=GPU())
# → Elements copied to GPU
# → Computation on GPU
# → Solution copied back to CPU
```

**User never sees CuArray!**

### 2. Backend Fallback

**Graceful degradation:**

```julia
# User requests GPU, but not available
sol = solve!(physics; backend=GPU())
# → Warning: "GPU requested but not available, falling back to CPU"
# → Uses CPU backend
```

**Prevents errors, maintains usability.**

### 3. Performance Hints

**Inform user about backend choice:**

```julia
sol = solve!(physics)
# Info: Using GPU backend (NVIDIA RTX 4090)
# Info: Transferred 1.2 GB to device
# Info: CG converged in 120 iterations (2.3s)
```

**User understands what happened without controlling it.**

## Migration Path

### Step 1: Rename (This Week)

- `GPUElasticityData` → `ElasticityDataGPU`
- `solve_elasticity_gpu!()` → `solve_backend!(::ElasticityDataGPU, ...)`
- Mark old names as `@deprecate`

### Step 2: Add CPU Backend (This Week)

- Create `ElasticityDataCPU`
- Implement `solve_backend!(::ElasticityDataCPU, ...)`
- Add `solve!()` with backend selection

### Step 3: Update Demos (This Week)

- Change `solve_elasticity_gpu!()` → `solve!()`
- Remove explicit CUDA checks
- Show automatic backend selection

### Step 4: Package Extension (Next Week)

- Move GPU code to `ext/JuliaFEMCUDAExt/`
- Test without CUDA installed
- Update documentation

## Conclusion

**Design principle:** Backend is **implementation detail**, not user concern.

**User code:**

```julia
sol = solve!(physics)  # ← Simple, portable, fast!
```

**Behind the scenes:**

- Automatic backend selection (GPU > CPU)
- Optimal performance for available hardware
- No user code changes needed

**Benefits:**

- ✅ Portable code (laptop → workstation → cluster)
- ✅ Clean API (no GPU types in user code)
- ✅ Performance transparency (automatic optimization)
- ✅ Gradual adoption (install CUDA → instant speedup)

This is the **Julia way** - multiple dispatch + package extensions = backend transparency!
