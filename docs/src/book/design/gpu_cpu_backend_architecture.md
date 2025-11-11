---
title: "GPU-First, CPU-Second Architecture"
date: 2025-11-10
status: "Design"
---

# Backend Architecture: GPU-First, CPU-Second

**Philosophy:** Write once, run on GPU or CPU. Maximum code reuse.

**Strategy:** Julia package extensions for backend-specific code.

## Design Goals

1. **Same API for GPU and CPU** - User code doesn't change
2. **Automatic fallback** - No CUDA? Use CPU automatically
3. **Maximum code reuse** - Shared physics, separate assembly
4. **GPU-first** - Best performance on GPU
5. **CPU-second** - Multithread fallback, not single-thread

## Architecture Overview

```
JuliaFEM.jl/
├── src/
│   ├── physics/
│   │   ├── physics_abstract.jl       # Abstract interface (shared)
│   │   ├── physics_elasticity.jl     # Physics{Elasticity} (shared)
│   │   └── physics_solvers.jl        # CG, Newton (backend-agnostic)
│   └── backends/
│       └── backend_interface.jl      # AbstractBackend, CPU(), GPU()
│
├── ext/
│   └── JuliaFEMCUDAExt/
│       ├── cuda_assembly.jl          # GPU kernels for assembly
│       ├── cuda_solvers.jl           # GPU CG solver
│       └── JuliaFEMCUDAExt.jl        # Extension entry point
│
└── Project.toml
    [weakdeps]
    CUDA = "..."
    [extensions]
    JuliaFEMCUDAExt = "CUDA"
```

## Package Extensions (Julia 1.9+)

**How it works:**

```toml
# Project.toml
[deps]
LinearAlgebra = "..."
SparseArrays = "..."

[weakdeps]
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"

[extensions]
JuliaFEMCUDAExt = "CUDA"
```

**Behavior:**

- **User has CUDA:** Extension loads automatically → GPU backend available
- **No CUDA:** Extension not loaded → CPU backend only
- **Seamless:** User code same either way

## Abstract Interface (Shared Code)

### Core Types

```julia
# src/physics/physics_abstract.jl

abstract type AbstractBackend end

struct CPU <: AbstractBackend
    nthreads::Int
end
CPU() = CPU(Threads.nthreads())

# GPU defined in extension (only if CUDA available)
# struct GPU <: AbstractBackend
#     device::CuDevice
# end

abstract type AbstractPhysics{P} end

# Physics{Elasticity} implementation
mutable struct Physics{P} <: AbstractPhysics{P}
    name::String
    dimension::Int
    properties::P
    
    # Geometry and material (backend-agnostic)
    body_elements::Vector{Element}
    
    # BCs (backend-agnostic representation)
    bc_dirichlet::DirichletBC
    bc_neumann::NeumannBC
    
    # Backend-specific data (union over concrete types)
    backend_data::Union{Nothing, AbstractBackendData}
end
```

### Boundary Conditions (Shared)

```julia
# src/physics/physics_elasticity.jl

struct DirichletBC
    node_ids::Vector{Int}
    components::Vector{Vector{Int}}
    values::Vector{Vector{Float64}}
end

struct NeumannBC
    surface_elements::Vector{Element}
    traction::Vector{Vec{3,Float64}}
end

# Add BCs (backend-agnostic)
function add_dirichlet!(
    physics::Physics{Elasticity},
    node_ids::Vector{Int},
    components::Vector{Int},
    value::Float64
)
    # Same for CPU and GPU
    bc = physics.bc_dirichlet
    for node in node_ids
        push!(bc.node_ids, node)
        push!(bc.components, components)
        push!(bc.values, fill(value, length(components)))
    end
end
```

### Solver Interface (Abstract)

```julia
# src/physics/physics_solvers.jl

"""
Main solver entry point - dispatches to backend
"""
function solve_physics!(
    physics::Physics{Elasticity};
    backend::AbstractBackend = default_backend(),
    time::Float64 = 0.0,
    tol = 1e-6,
    max_iter = 1000
)
    # Initialize backend data
    if physics.backend_data === nothing
        physics.backend_data = initialize_backend!(physics, backend, time)
    end
    
    # Dispatch to backend-specific solver
    return solve_backend!(physics, backend, time, tol, max_iter)
end

# Default: Use GPU if available, else CPU
function default_backend()
    if @isdefined(CUDA) && CUDA.functional()
        return GPU()
    else
        return CPU()
    end
end
```

## CPU Backend (Core - No Extension Needed)

### Assembly with Multithreading

```julia
# src/backends/cpu_assembly.jl

struct CPUBackendData <: AbstractBackendData
    # Global arrays (CPU)
    nodes::Matrix{Float64}              # 3 × n_nodes
    elements::Matrix{Int}               # 4 × n_elements
    E::Vector{Float64}
    ν::Vector{Float64}
    
    # BC arrays
    is_fixed::Vector{Bool}
    prescribed::Vector{Float64}
    
    # Surface loads
    surface_nodes::Matrix{Int}
    surface_traction::Matrix{Float64}
    
    # Working arrays
    f_ext::Vector{Float64}
    u::Vector{Float64}
    
    # Threading data
    element_stresses::Vector{Vector{SymmetricTensor{2,3,Float64,6}}}  # Per thread
end

"""
Compute residual using multithreading
"""
function compute_residual_cpu!(
    data::CPUBackendData,
    u::Vector{Float64}
)
    n_threads = Threads.nthreads()
    n_elements = size(data.elements, 2)
    
    # Phase 1: Compute stresses (parallel over elements)
    stresses = data.element_stresses
    
    Threads.@threads for elem_idx in 1:n_elements
        tid = Threads.threadid()
        
        # Extract element data
        conn = data.elements[:, elem_idx]
        X = data.nodes[:, conn]
        u_elem = [u[3*n-2:3*n] for n in conn]
        E = data.E[elem_idx]
        ν = data.ν[elem_idx]
        
        # Compute stress (same as GPU kernel, but CPU)
        σ = compute_element_stress(X, u_elem, E, ν)
        stresses[tid][elem_idx] = σ
    end
    
    # Phase 2: Nodal assembly (parallel over nodes)
    n_nodes = size(data.nodes, 2)
    f_int = zeros(3 * n_nodes)
    
    # Node-to-elements map needed (built during initialization)
    Threads.@threads for node_idx in 1:n_nodes
        f_node = zero(Vec{3,Float64})
        
        # Loop over elements touching this node
        for elem_idx in node_to_elements[node_idx]
            # Accumulate force from this element
            f_node += compute_nodal_force(node_idx, elem_idx, stresses, data)
        end
        
        f_int[3*node_idx-2:3*node_idx] = [f_node[1], f_node[2], f_node[3]]
    end
    
    # Residual
    r = f_int - data.f_ext
    
    # Apply Dirichlet BC
    r[data.is_fixed] .= 0.0
    
    return r
end
```

### CPU CG Solver

```julia
# src/backends/cpu_solvers.jl

function cg_solve_cpu!(
    data::CPUBackendData;
    tol = 1e-6,
    max_iter = 1000
)
    n_dofs = length(data.f_ext)
    u = data.u
    b = data.f_ext
    
    # Initial residual
    r = b - compute_residual_cpu!(data, u)
    r[data.is_fixed] .= 0.0
    
    p = copy(r)
    r_dot_r = dot(r, r)
    
    for iter in 1:max_iter
        # Matrix-free: Ap = K*p
        Ap = compute_residual_cpu!(data, p)
        Ap[data.is_fixed] .= 0.0
        
        alpha = r_dot_r / dot(p, Ap)
        u .+= alpha .* p
        r .-= alpha .* Ap
        
        r_dot_r_new = dot(r, r)
        
        if sqrt(r_dot_r_new) < tol
            @info "  CPU CG converged in $iter iterations"
            return iter, sqrt(r_dot_r_new)
        end
        
        beta = r_dot_r_new / r_dot_r
        p .= r .+ beta .* p
        r_dot_r = r_dot_r_new
    end
    
    @warn "CPU CG did not converge"
    return max_iter, sqrt(r_dot_r)
end
```

## GPU Backend (Extension)

### Extension Structure

```julia
# ext/JuliaFEMCUDAExt/JuliaFEMCUDAExt.jl

module JuliaFEMCUDAExt

using JuliaFEM
using CUDA
using Tensors

# Define GPU backend type
struct GPU <: JuliaFEM.AbstractBackend
    device::CuDevice
end
GPU() = GPU(CuDevice(0))

# Export to make available when extension loads
export GPU

# GPU-specific data
struct GPUBackendData <: JuliaFEM.AbstractBackendData
    nodes::CuArray{Float64,2}
    elements::CuArray{Int32,2}
    E::CuArray{Float64,1}
    ν::CuArray{Float64,1}
    is_fixed::CuArray{Bool,1}
    prescribed::CuArray{Float64,1}
    surface_nodes::CuArray{Int32,2}
    surface_traction::CuArray{Float64,2}
    f_ext::CuArray{Float64,1}
    u::CuArray{Float64,1}
    # ... node-to-elements map
end

# Include GPU assembly and solvers
include("cuda_assembly.jl")
include("cuda_solvers.jl")

# Register backend
function JuliaFEM.initialize_backend!(
    physics::Physics{Elasticity},
    backend::GPU,
    time::Float64
)
    # Transfer CPU data to GPU
    return initialize_gpu_data(physics, time)
end

function JuliaFEM.solve_backend!(
    physics::Physics{Elasticity},
    backend::GPU,
    time, tol, max_iter
)
    gpu_data = physics.backend_data::GPUBackendData
    
    # Compute external forces
    compute_external_forces_gpu!(gpu_data)
    
    # Solve
    iterations, residual = cg_solve_gpu!(gpu_data; tol, max_iter)
    
    # Return CPU array
    u_cpu = Array(gpu_data.u)
    return (u=u_cpu, iterations=iterations, residual=residual)
end

end # module
```

### GPU Assembly (Same Kernels)

```julia
# ext/JuliaFEMCUDAExt/cuda_assembly.jl

# Same CUDA kernels as before
function compute_element_stresses_kernel!(...)
    # ... exact same as gpu_physics_elasticity.jl
end

function nodal_assembly_kernel!(...)
    # ... exact same as gpu_physics_elasticity.jl
end

function apply_surface_traction_kernel!(...)
    # ... exact same as gpu_physics_elasticity.jl
end

function apply_dirichlet_kernel!(...)
    # ... exact same as gpu_physics_elasticity.jl
end

function compute_residual_gpu!(gpu_data::GPUBackendData, u::CuArray)
    # ... same implementation
end
```

## User API (Unified)

### Example: Cantilever Beam

```julia
using JuliaFEM

# Optional: Load CUDA support
using CUDA  # If installed, GPU backend available

# Create elements (same for CPU and GPU)
body_elements = Element[]
for conn in connectivity
    el = Element(Tet4, conn)
    update!(el, "geometry", 0.0 => X)
    update!(el, "youngs modulus", 0.0 => 210e9)
    update!(el, "poissons ratio", 0.0 => 0.3)
    push!(body_elements, el)
end

# Create physics (same for CPU and GPU)
physics = Physics(Elasticity, "cantilever", 3)
add_elements!(physics, body_elements)

# Add BCs (same for CPU and GPU)
add_dirichlet!(physics, [1, 2, 3], [1,2,3], 0.0)

surf = Element(Tri3, [4, 5, 6])
update!(surf, "geometry", 0.0 => X_surf)
add_neumann!(physics, surf, Vec{3}((0.0, 0.0, -1e6)))

# Solve - automatic backend selection
result = solve_physics!(physics)  # Uses GPU if available, else CPU

# Or explicit backend
result = solve_physics!(physics, backend=GPU())      # Force GPU
result = solve_physics!(physics, backend=CPU(8))     # Force CPU with 8 threads
```

### Backend Selection

```julia
# Automatic (default)
result = solve_physics!(physics)

# Manual
if CUDA.functional()
    result = solve_physics!(physics, backend=GPU())
    println("Solved on GPU")
else
    result = solve_physics!(physics, backend=CPU())
    println("Solved on CPU with $(Threads.nthreads()) threads")
end
```

## Code Reuse Analysis

### Shared (80% of code)

**✅ Physics definition:**
- `Physics{Elasticity}` type
- `add_elements!`
- `add_dirichlet!`, `add_neumann!`
- BC data structures

**✅ Solver logic:**
- CG algorithm (same structure)
- Convergence checks
- Newton iteration loop (future)

**✅ Material models:**
- Hooke's law
- Plasticity algorithms
- Stress update

**✅ Integration:**
- Gauss points
- Shape functions
- Jacobian computation

### Backend-Specific (20% of code)

**GPU (in extension):**
- CUDA kernel launches (`@cuda`)
- CuArray operations
- GPU memory management

**CPU (in core):**
- `Threads.@threads` loops
- Regular Array operations
- Thread-local storage

## Performance Characteristics

### GPU Backend

**Strengths:**
- Massive parallelism (1000+ threads)
- Matrix-free (no memory for K)
- Fast for large problems (100K+ DOFs)

**Weaknesses:**
- Small problems: kernel launch overhead
- CPU-GPU transfer (if not careful)

**Best for:** > 10K DOFs

### CPU Backend

**Strengths:**
- No CPU-GPU transfer
- Good single-element performance
- Easier debugging

**Weaknesses:**
- Limited parallelism (typ. 8-16 threads)
- Needs more memory (thread-local arrays)

**Best for:** < 10K DOFs, or no GPU available

### Crossover Point (Projected)

| DOFs | GPU Time | CPU Time (8 threads) | Winner |
|------|----------|----------------------|--------|
| 1K | 50 ms | 20 ms | CPU |
| 10K | 200 ms | 200 ms | TIE |
| 100K | 2 s | 20 s | GPU |
| 1M | 20 s | 200 s | GPU |

## Implementation Plan

### Phase 1: Refactor Current Code (Day 1)

**Goal:** Extract shared interface from `gpu_physics_elasticity.jl`

**Tasks:**

1. Create `src/physics/` directory
2. Move `Physics{Elasticity}` to `physics_elasticity.jl` (no CUDA imports)
3. Move BC structs to shared file
4. Define `AbstractBackend`, `AbstractBackendData`

### Phase 2: CPU Backend (Day 2)

**Goal:** Multithreaded CPU implementation

**Tasks:**

1. Create `src/backends/cpu_assembly.jl`
2. Implement `compute_residual_cpu!` with `Threads.@threads`
3. Implement `cg_solve_cpu!`
4. Test on cantilever beam

### Phase 3: Extension Setup (Day 3)

**Goal:** Move GPU code to extension

**Tasks:**

1. Create `ext/JuliaFEMCUDAExt/` directory
2. Update `Project.toml` with weakdeps and extensions
3. Move GPU kernels to extension
4. Implement `initialize_backend!` and `solve_backend!`
5. Test with and without CUDA

### Phase 4: Unified API (Day 4)

**Goal:** Single entry point for both backends

**Tasks:**

1. Implement `solve_physics!` dispatch
2. Automatic backend detection
3. Update demos to use unified API
4. Document backend selection

### Phase 5: Optimization (Week 2)

**Goal:** Performance tuning

**Tasks:**

1. Benchmark CPU vs GPU
2. Optimize threading (CPU)
3. Optimize kernel parameters (GPU)
4. Add preconditioning (both backends)

## Testing Strategy

### Unit Tests

```julia
@testset "Physics Interface" begin
    physics = Physics(Elasticity, "test", 3)
    # ... test add_elements!, add_dirichlet!, etc.
end

@testset "CPU Backend" begin
    physics = setup_test_problem()
    result = solve_physics!(physics, backend=CPU(1))
    @test result.residual < 1e-6
end

@testset "GPU Backend" begin
    if CUDA.functional()
        physics = setup_test_problem()
        result = solve_physics!(physics, backend=GPU())
        @test result.residual < 1e-6
    else
        @test_skip "CUDA not available"
    end
end
```

### Integration Tests

```julia
@testset "Cantilever CPU vs GPU" begin
    physics = setup_cantilever()
    
    result_cpu = solve_physics!(physics, backend=CPU())
    
    if CUDA.functional()
        result_gpu = solve_physics!(physics, backend=GPU())
        
        # Solutions should match
        @test isapprox(result_cpu.u, result_gpu.u, rtol=1e-6)
    end
end
```

### Performance Tests

```julia
using BenchmarkTools

physics = setup_large_problem(n_nodes=100_000)

# CPU
@btime solve_physics!($physics, backend=CPU())

# GPU
if CUDA.functional()
    @btime solve_physics!($physics, backend=GPU())
end
```

## Migration Path

### From Current Code

**Old (GPU only):**
```julia
include("src/gpu_physics_elasticity.jl")
using .GPUElasticityPhysics

physics = Physics(Elasticity, "body", 3)
result = solve_elasticity_gpu!(physics)
```

**New (GPU or CPU):**
```julia
using JuliaFEM

physics = Physics(Elasticity, "body", 3)
result = solve_physics!(physics)  # Automatic backend
```

**Breaking changes:**
- `solve_elasticity_gpu!` → `solve_physics!`
- Must specify backend explicitly if want GPU-only: `backend=GPU()`

## Benefits

**For Users:**

✅ **Automatic fallback** - No CUDA? No problem, uses CPU  
✅ **Same API** - Write once, run anywhere  
✅ **Explicit control** - Can force backend if desired  
✅ **No dependencies** - CUDA optional, not required

**For Developers:**

✅ **Code reuse** - 80% shared between backends  
✅ **Maintainability** - One physics implementation  
✅ **Testability** - Can test CPU without GPU  
✅ **Extensibility** - Easy to add new backends (Metal, ROCm, etc.)

## Future Backends

**Potential additions:**

- **Metal GPU** (Apple Silicon) - Same extension pattern
- **ROCm** (AMD GPU) - Same extension pattern
- **Distributed CPU** (MPI) - Use same Physics type
- **Intel oneAPI** (Intel GPUs) - Same extension pattern

**All use same `AbstractBackend` interface!**

## File Structure (Final)

```
JuliaFEM.jl/
├── src/
│   ├── JuliaFEM.jl                   # Main module
│   ├── physics/
│   │   ├── physics_abstract.jl       # AbstractPhysics, AbstractBackend
│   │   ├── physics_elasticity.jl     # Physics{Elasticity} (shared)
│   │   ├── physics_solvers.jl        # solve_physics! dispatch
│   │   └── physics_bcs.jl            # BC types (shared)
│   └── backends/
│       ├── backend_interface.jl      # AbstractBackendData, CPU type
│       ├── cpu_assembly.jl           # CPU multithreaded assembly
│       └── cpu_solvers.jl            # CPU CG solver
│
├── ext/
│   └── JuliaFEMCUDAExt/
│       ├── JuliaFEMCUDAExt.jl        # Extension entry, GPU type
│       ├── cuda_assembly.jl          # GPU kernels
│       └── cuda_solvers.jl           # GPU CG solver
│
├── demos/
│   ├── cantilever_unified.jl         # Works on CPU or GPU
│   └── backend_comparison.jl         # Benchmark both
│
├── test/
│   ├── test_physics_interface.jl     # Backend-agnostic tests
│   ├── test_cpu_backend.jl           # CPU-specific tests
│   └── test_gpu_backend.jl           # GPU-specific tests (skip if no CUDA)
│
└── Project.toml
    [deps]
    LinearAlgebra = "..."
    Tensors = "..."
    
    [weakdeps]
    CUDA = "..."
    
    [extensions]
    JuliaFEMCUDAExt = "CUDA"
```

## Summary

**Key Insights:**

1. **GPU-first, CPU-second** - Best performance on GPU, reliable fallback on CPU
2. **Package extensions** - CUDA optional, loads automatically if available
3. **Maximum code reuse** - 80% shared, 20% backend-specific
4. **Same user API** - Write once, run on GPU or CPU
5. **Explicit control** - Can force backend if needed

**Next Steps:**

1. Refactor current GPU code to extract shared interface
2. Implement CPU backend with multithreading
3. Set up package extension for GPU
4. Update demos to use unified API
5. Benchmark and optimize

**Result:** Users get best of both worlds - GPU performance when available, CPU fallback always works.
