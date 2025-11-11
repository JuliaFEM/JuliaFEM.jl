---
title: "Migration Guide: GPU-Only to GPU-First/CPU-Second"
date: 2025-11-10
status: "Implementation Plan"
---

# Migration Guide: Adding CPU Backend

**Goal:** Refactor `src/gpu_physics_elasticity.jl` to support both GPU and CPU with maximum code reuse.

**Strategy:** Extract shared interface, implement CPU backend, move GPU to extension.

## Step-by-Step Migration

### Step 1: Extract Shared Types (Day 1, Morning)

**Create:** `src/physics/physics_types.jl`

**Extract from `gpu_physics_elasticity.jl`:**

```julia
# Shared types (no CUDA dependency)
abstract type AbstractBackend end
abstract type AbstractBackendData end

struct CPU <: AbstractBackend
    nthreads::Int
end
CPU() = CPU(Threads.nthreads())

# Elasticity properties (no CUDA)
mutable struct Elasticity <: FieldProblem
    formulation::Symbol
    finite_strain::Bool
    geometric_stiffness::Bool
end

# BC types (no CUDA)
struct DirichletBC
    node_ids::Vector{Int}
    components::Vector{Vector{Int}}
    values::Vector{Vector{Float64}}
end

struct NeumannBC
    surface_elements::Vector{Element}
    traction::Vector{Vec{3,Float64}}
end

# Physics type (backend-agnostic)
mutable struct Physics{P} <: AbstractPhysics{P}
    name::String
    dimension::Int
    properties::P
    body_elements::Vector{Element}
    bc_dirichlet::DirichletBC
    bc_neumann::NeumannBC
    backend_data::Union{Nothing, AbstractBackendData}
end
```

**No changes to:**
- `add_elements!`
- `add_dirichlet!`
- `add_neumann!`

These work for both CPU and GPU!

### Step 2: Create Backend Interface (Day 1, Afternoon)

**Create:** `src/physics/physics_interface.jl`

```julia
"""
Initialize backend data from Physics
"""
function initialize_backend! end

"""
Solve physics on specific backend
"""
function solve_backend! end

"""
Main entry point - dispatches to backend
"""
function solve_physics!(
    physics::Physics{Elasticity};
    backend::AbstractBackend = auto_backend(),
    time::Float64 = 0.0,
    tol = 1e-6,
    max_iter = 1000
)
    # Initialize if needed
    if physics.backend_data === nothing
        physics.backend_data = initialize_backend!(physics, backend, time)
    end
    
    # Dispatch to backend
    return solve_backend!(physics, backend, time, tol, max_iter)
end

"""
Auto-detect best backend
"""
function auto_backend()
    # Check if GPU extension loaded
    if isdefined(Main, :CUDA) && Main.CUDA.functional()
        return GPU()  # Defined in extension
    else
        return CPU()
    end
end
```

### Step 3: Implement CPU Backend (Day 2)

**Create:** `src/backends/cpu_backend.jl`

#### 3a. CPU Data Structure

```julia
struct CPUBackendData <: AbstractBackendData
    # Geometry
    nodes::Matrix{Float64}              # 3 × n_nodes
    elements::Matrix{Int32}             # 4 × n_elements
    n_nodes::Int
    n_elements::Int
    
    # Material
    E::Vector{Float64}
    ν::Vector{Float64}
    
    # BCs
    is_fixed::Vector{Bool}
    prescribed::Vector{Float64}
    
    # Surface loads
    surface_nodes::Matrix{Int32}
    surface_traction::Matrix{Float64}
    
    # Node-to-elements (CSR)
    node_to_elem_ptr::Vector{Int32}
    node_to_elem_data::Vector{Int32}
    
    # Working arrays
    f_ext::Vector{Float64}
    u::Vector{Float64}
end
```

#### 3b. Initialize CPU Backend

```julia
function initialize_backend!(
    physics::Physics{Elasticity},
    backend::CPU,
    time::Float64
)
    @info "Initializing CPU backend ($(backend.nthreads) threads)..."
    
    # Extract data from elements (same as GPU version)
    n_elements = length(physics.body_elements)
    
    # Build node map
    node_set = Set{Int}()
    for el in physics.body_elements
        for node in get_connectivity(el)
            push!(node_set, node)
        end
    end
    node_list = sort(collect(node_set))
    node_map = Dict(node => i for (i, node) in enumerate(node_list))
    n_nodes = length(node_list)
    
    # Extract coordinates
    nodes = zeros(3, n_nodes)
    for el in physics.body_elements
        X = el("geometry", time)
        conn = get_connectivity(el)
        for (local_idx, global_node) in enumerate(conn)
            renumbered = node_map[global_node]
            nodes[:, renumbered] = X[:, local_idx]
        end
    end
    
    # Build connectivity
    elements = zeros(Int32, 4, n_elements)
    for (i, el) in enumerate(physics.body_elements)
        conn = get_connectivity(el)
        for (j, node) in enumerate(conn)
            elements[j, i] = node_map[node]
        end
    end
    
    # Extract materials
    E = [el("youngs modulus", time) for el in physics.body_elements]
    ν = [el("poissons ratio", time) for el in physics.body_elements]
    
    # Build BC arrays (same as GPU)
    n_dofs = 3 * n_nodes
    is_fixed = fill(false, n_dofs)
    prescribed = zeros(n_dofs)
    
    bc = physics.bc_dirichlet
    for (i, node_id) in enumerate(bc.node_ids)
        if haskey(node_map, node_id)
            renumbered = node_map[node_id]
            for (comp_idx, comp) in enumerate(bc.components[i])
                dof = 3 * (renumbered - 1) + comp
                is_fixed[dof] = true
                prescribed[dof] = bc.values[i][comp_idx]
            end
        end
    end
    
    # Surface loads (same as GPU)
    n_surface = length(physics.bc_neumann.surface_elements)
    surface_nodes = zeros(Int32, 3, n_surface)
    surface_traction = zeros(Float64, 3, n_surface)
    
    for (i, surf_el) in enumerate(physics.bc_neumann.surface_elements)
        conn = get_connectivity(surf_el)
        for (j, node) in enumerate(conn)
            surface_nodes[j, i] = node_map[node]
        end
        t = physics.bc_neumann.traction[i]
        surface_traction[:, i] = [t[1], t[2], t[3]]
    end
    
    # Node-to-elements map
    node_to_elems = [Int32[] for _ in 1:n_nodes]
    for (el_idx, el) in enumerate(physics.body_elements)
        for node in get_connectivity(el)
            renumbered = node_map[node]
            push!(node_to_elems[renumbered], el_idx)
        end
    end
    
    ptr = zeros(Int32, n_nodes + 1)
    ptr[1] = 1
    for i in 1:n_nodes
        ptr[i+1] = ptr[i] + length(node_to_elems[i])
    end
    data = vcat(node_to_elems...)
    
    # Create backend data
    return CPUBackendData(
        nodes, elements, n_nodes, n_elements,
        E, ν,
        is_fixed, prescribed,
        surface_nodes, surface_traction,
        ptr, data,
        zeros(n_dofs), zeros(n_dofs)
    )
end
```

#### 3c. CPU Assembly Functions

```julia
"""
Compute stress at one Gauss point (pure function, no threading)
"""
function compute_stress_at_gp(
    X::NTuple{4,Vec{3}},
    u::NTuple{4,Vec{3}},
    E::Float64,
    ν::Float64
)
    # Shape derivatives (Tet4)
    dN_dxi = (
        Vec{3}((-1.0, -1.0, -1.0)),
        Vec{3}((1.0, 0.0, 0.0)),
        Vec{3}((0.0, 1.0, 0.0)),
        Vec{3}((0.0, 0.0, 1.0))
    )
    
    # Jacobian
    J = dN_dxi[1] ⊗ X[1] + dN_dxi[2] ⊗ X[2] + 
        dN_dxi[3] ⊗ X[3] + dN_dxi[4] ⊗ X[4]
    invJ = inv(J)
    
    # Physical derivatives
    dN_dx = (invJ ⋅ dN_dxi[1], invJ ⋅ dN_dxi[2], 
             invJ ⋅ dN_dxi[3], invJ ⋅ dN_dxi[4])
    
    # Strain
    ε = symmetric(dN_dx[1] ⊗ u[1] + dN_dx[2] ⊗ u[2] + 
                  dN_dx[3] ⊗ u[3] + dN_dx[4] ⊗ u[4])
    
    # Stress (Hooke)
    λ = E * ν / ((1 + ν) * (1 - 2ν))
    μ = E / (2(1 + ν))
    I = one(ε)
    σ = λ * tr(ε) * I + 2μ * ε
    
    return σ, det(J)
end

"""
Compute residual with multithreading
"""
function compute_residual_cpu!(
    data::CPUBackendData,
    u::Vector{Float64}
)
    n_nodes = data.n_nodes
    n_elements = data.n_elements
    
    # Phase 1: Internal forces (nodal assembly, parallel over nodes)
    f_int = zeros(3 * n_nodes)
    
    Threads.@threads for node_idx in 1:n_nodes
        f_node = zero(Vec{3,Float64})
        gauss_weight = 1.0 / 24.0  # Tet4
        
        # Loop over elements touching this node
        elem_start = data.node_to_elem_ptr[node_idx]
        elem_end = data.node_to_elem_ptr[node_idx+1] - 1
        
        for elem_offset in elem_start:elem_end
            elem_idx = data.node_to_elem_data[elem_offset]
            
            # Get element data
            conn = data.elements[:, elem_idx]
            local_node = findfirst(==(node_idx), conn)
            
            # Coordinates
            X = tuple([Vec{3}((data.nodes[1, n], data.nodes[2, n], data.nodes[3, n]))
                      for n in conn]...)
            
            # Displacements
            u_elem = tuple([Vec{3}((u[3*n-2], u[3*n-1], u[3*n]))
                           for n in conn]...)
            
            # Material
            E = data.E[elem_idx]
            ν = data.ν[elem_idx]
            
            # Compute stress and det(J)
            σ, detJ = compute_stress_at_gp(X, u_elem, E, ν)
            
            # Shape derivative for this node
            dN_dxi = [Vec{3}((-1.0, -1.0, -1.0)), Vec{3}((1.0, 0.0, 0.0)),
                     Vec{3}((0.0, 1.0, 0.0)), Vec{3}((0.0, 0.0, 1.0))][local_node]
            J = dN_dxi ⊗ X[1] + ...  # Recompute (or cache)
            invJ = inv(J)
            dN_dx = invJ ⋅ dN_dxi
            
            # Accumulate force (4 Gauss points, but Tet4 constant stress)
            f_node += (dN_dx ⋅ σ) * (4 * gauss_weight * detJ)
        end
        
        f_int[3*node_idx-2] = f_node[1]
        f_int[3*node_idx-1] = f_node[2]
        f_int[3*node_idx] = f_node[3]
    end
    
    # Residual
    r = f_int - data.f_ext
    
    # Apply Dirichlet
    r[data.is_fixed] .= 0.0
    
    return r
end
```

#### 3d. CPU CG Solver

```julia
function cg_solve_cpu!(
    data::CPUBackendData;
    tol = 1e-6,
    max_iter = 1000
)
    u = data.u
    b = data.f_ext
    
    # Initial residual
    r = b - compute_residual_cpu!(data, u)
    r[data.is_fixed] .= 0.0
    
    p = copy(r)
    r_dot_r = dot(r, r)
    
    for iter in 1:max_iter
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

#### 3e. CPU Solver Entry Point

```julia
function solve_backend!(
    physics::Physics{Elasticity},
    backend::CPU,
    time::Float64,
    tol,
    max_iter
)
    data = physics.backend_data::CPUBackendData
    
    # Compute external forces (Neumann BC)
    @info "Computing external forces (CPU)..."
    fill!(data.f_ext, 0.0)
    
    n_surface = size(data.surface_nodes, 2)
    for surf_idx in 1:n_surface
        # Get nodes
        n1, n2, n3 = data.surface_nodes[:, surf_idx]
        
        # Coordinates
        X1 = Vec{3}((data.nodes[1, n1], data.nodes[2, n1], data.nodes[3, n1]))
        X2 = Vec{3}((data.nodes[1, n2], data.nodes[2, n2], data.nodes[3, n2]))
        X3 = Vec{3}((data.nodes[1, n3], data.nodes[2, n3], data.nodes[3, n3]))
        
        # Area
        area = 0.5 * norm((X2 - X1) × (X3 - X1))
        
        # Traction
        t = Vec{3}((data.surface_traction[1, surf_idx],
                   data.surface_traction[2, surf_idx],
                   data.surface_traction[3, surf_idx]))
        
        # Distribute to nodes
        force = (area / 3.0) * t
        data.f_ext[3*n1-2:3*n1] .+= [force[1], force[2], force[3]]
        data.f_ext[3*n2-2:3*n2] .+= [force[1], force[2], force[3]]
        data.f_ext[3*n3-2:3*n3] .+= [force[1], force[2], force[3]]
    end
    
    # Solve
    @info "Solving with CPU CG ($(backend.nthreads) threads)..."
    fill!(data.u, 0.0)
    iterations, residual = cg_solve_cpu!(data; tol, max_iter)
    
    return (u=copy(data.u), iterations=iterations, residual=residual)
end
```

### Step 4: Move GPU to Extension (Day 3)

**Create:** `ext/JuliaFEMCUDAExt/`

#### 4a. Extension Entry Point

**File:** `ext/JuliaFEMCUDAExt/JuliaFEMCUDAExt.jl`

```julia
module JuliaFEMCUDAExt

using JuliaFEM
using CUDA
using Tensors

# Define GPU backend
struct GPU <: JuliaFEM.AbstractBackend
    device::CuDevice
end
GPU() = GPU(CUDA.device())

export GPU

# Include GPU-specific code
include("gpu_backend.jl")
include("gpu_kernels.jl")

end
```

#### 4b. GPU Backend Data

**File:** `ext/JuliaFEMCUDAExt/gpu_backend.jl`

```julia
struct GPUBackendData <: JuliaFEM.AbstractBackendData
    # Same fields as CPUBackendData, but CuArrays
    nodes::CuArray{Float64,2}
    elements::CuArray{Int32,2}
    # ... (copy from current gpu_physics_elasticity.jl)
end

# initialize_backend! for GPU
# (copy from current code, rename GPUElasticityData → GPUBackendData)

# solve_backend! for GPU
# (copy from current code)
```

#### 4c. GPU Kernels

**File:** `ext/JuliaFEMCUDAExt/gpu_kernels.jl`

```julia
# Copy all CUDA kernels from current gpu_physics_elasticity.jl:
# - compute_element_stresses_kernel!
# - nodal_assembly_kernel!
# - apply_surface_traction_kernel!
# - apply_dirichlet_kernel!
# - compute_residual_gpu!
# - cg_solve_gpu!
```

### Step 5: Update Project.toml (Day 3)

```toml
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
Tensors = "48a634ad-e948-5137-8d70-aa71f2a747f4"

[weakdeps]
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"

[extensions]
JuliaFEMCUDAExt = "CUDA"

[compat]
julia = "1.9"
CUDA = "5"
```

### Step 6: Update Demos (Day 4)

**Create:** `demos/cantilever_unified.jl`

```julia
using JuliaFEM

# Try to load CUDA
try
    using CUDA
    println("CUDA available: $(CUDA.functional())")
catch
    println("CUDA not available, using CPU")
end

# Create physics (same for CPU or GPU)
physics = Physics(Elasticity, "cantilever", 3)
# ... add elements, BCs

# Solve - automatic backend
result = solve_physics!(physics)
println("Solved with automatic backend")

# Or explicit
if @isdefined(CUDA) && CUDA.functional()
    result_gpu = solve_physics!(physics, backend=GPU())
    println("GPU result: $(result_gpu.iterations) iterations")
end

result_cpu = solve_physics!(physics, backend=CPU())
println("CPU result: $(result_cpu.iterations) iterations")

# Compare
if @isdefined(result_gpu)
    diff = norm(result_gpu.u - result_cpu.u)
    println("GPU vs CPU difference: $diff")
end
```

## File Reorganization

### Before (Current)

```
src/
├── JuliaFEM.jl
└── gpu_physics_elasticity.jl  (716 lines, all GPU)
```

### After (Target)

```
src/
├── JuliaFEM.jl
├── physics/
│   ├── physics_types.jl          # Shared types (100 lines)
│   ├── physics_interface.jl      # solve_physics! (50 lines)
│   └── physics_elasticity.jl     # add_elements!, add_dirichlet! (100 lines)
└── backends/
    └── cpu_backend.jl            # CPU implementation (400 lines)

ext/
└── JuliaFEMCUDAExt/
    ├── JuliaFEMCUDAExt.jl        # Extension entry (20 lines)
    ├── gpu_backend.jl            # GPU data structures (100 lines)
    └── gpu_kernels.jl            # CUDA kernels (400 lines)
```

**Line count:**
- **Shared:** 250 lines (types + interface + elasticity)
- **CPU:** 400 lines (backend + assembly + CG)
- **GPU:** 520 lines (extension + backend + kernels)
- **Total:** 1170 lines (vs 716 before, but now supports CPU!)

## Testing Plan

### Phase 1: Test Extraction (Day 4)

Test that shared types work:

```julia
@testset "Shared Types" begin
    physics = Physics(Elasticity, "test", 3)
    @test physics.dimension == 3
    
    add_dirichlet!(physics, [1], [1,2,3], 0.0)
    @test length(physics.bc_dirichlet.node_ids) == 1
end
```

### Phase 2: Test CPU Backend (Day 4)

```julia
@testset "CPU Backend" begin
    physics = setup_simple_problem()
    result = solve_physics!(physics, backend=CPU(1))
    @test result.residual < 1e-6
end

@testset "CPU Multithreading" begin
    physics = setup_simple_problem()
    result = solve_physics!(physics, backend=CPU(4))
    @test result.residual < 1e-6
    @test result.iterations < 500
end
```

### Phase 3: Test GPU Extension (Day 5)

```julia
@testset "GPU Backend" begin
    if @isdefined(CUDA) && CUDA.functional()
        physics = setup_simple_problem()
        result = solve_physics!(physics, backend=GPU())
        @test result.residual < 1e-6
    else
        @test_skip "CUDA not available"
    end
end

@testset "CPU vs GPU" begin
    if @isdefined(CUDA) && CUDA.functional()
        physics = setup_simple_problem()
        
        result_cpu = solve_physics!(physics, backend=CPU())
        result_gpu = solve_physics!(physics, backend=GPU())
        
        @test isapprox(result_cpu.u, result_gpu.u, rtol=1e-6)
    end
end
```

## Checklist

### Day 1: Extract Shared Code

- [ ] Create `src/physics/physics_types.jl`
- [ ] Move `Elasticity`, `DirichletBC`, `NeumannBC`, `Physics{T}`
- [ ] Create `AbstractBackend`, `CPU` type
- [ ] Test that types work without CUDA

### Day 2: Implement CPU Backend

- [ ] Create `src/backends/cpu_backend.jl`
- [ ] Implement `CPUBackendData`
- [ ] Implement `initialize_backend!(physics, CPU(), time)`
- [ ] Implement `compute_residual_cpu!` with threading
- [ ] Implement `cg_solve_cpu!`
- [ ] Implement `solve_backend!(physics, CPU(), ...)`
- [ ] Test CPU solve on simple problem

### Day 3: Create Extension

- [ ] Create `ext/JuliaFEMCUDAExt/` directory
- [ ] Update `Project.toml` with weakdeps
- [ ] Create `JuliaFEMCUDAExt.jl` entry point
- [ ] Define `GPU` backend type
- [ ] Move GPU kernels to `gpu_kernels.jl`
- [ ] Implement `initialize_backend!(physics, GPU(), time)`
- [ ] Implement `solve_backend!(physics, GPU(), ...)`
- [ ] Test GPU solve with extension

### Day 4: Integration

- [ ] Create `solve_physics!` dispatch
- [ ] Implement `auto_backend()` selection
- [ ] Update `src/JuliaFEM.jl` to include new files
- [ ] Remove old `gpu_physics_elasticity.jl` (or deprecate)
- [ ] Create `demos/cantilever_unified.jl`
- [ ] Test automatic backend selection

### Day 5: Documentation & Testing

- [ ] Update user manual with new API
- [ ] Document backend selection
- [ ] Write CPU backend tests
- [ ] Write GPU extension tests
- [ ] Write comparison tests (CPU vs GPU)
- [ ] Benchmark both backends

## Expected Results

### User Experience

**Without CUDA:**
```julia
julia> using JuliaFEM
julia> physics = Physics(Elasticity, "body", 3)
julia> result = solve_physics!(physics)
[ Info: Initializing CPU backend (8 threads)...
[ Info: CPU CG converged in 430 iterations
```

**With CUDA:**
```julia
julia> using JuliaFEM, CUDA
julia> physics = Physics(Elasticity, "body", 3)
julia> result = solve_physics!(physics)
[ Info: Initializing GPU backend...
[ Info: GPU CG converged in 430 iterations
```

### Performance (Projected)

| Problem Size | CPU (8 threads) | GPU | Speedup |
|--------------|-----------------|-----|---------|
| 1K DOFs | 50 ms | 100 ms | 0.5× |
| 10K DOFs | 500 ms | 300 ms | 1.7× |
| 100K DOFs | 50 s | 3 s | 17× |
| 1M DOFs | 500 s | 20 s | 25× |

## Summary

**Migration strategy:**

1. ✅ **Extract** shared types (Physics, BCs) - no CUDA dependency
2. ✅ **Implement** CPU backend with multithreading
3. ✅ **Move** GPU code to package extension
4. ✅ **Unify** API with `solve_physics!(physics, backend)`
5. ✅ **Test** both backends, compare results

**Result:**

- **Same API** for CPU and GPU
- **Automatic fallback** if no CUDA
- **Maximum code reuse** (250 lines shared)
- **GPU-first** (best performance)
- **CPU-second** (reliable fallback)

**Time estimate:** 5 days full-time work
