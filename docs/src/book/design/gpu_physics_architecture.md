---
title: "GPU Physics Architecture"
date: 2025-11-10
status: "Implemented"
---

# GPU-First Physics{Elasticity} Architecture

**Status:** ✅ Implemented (November 10, 2025)

## Overview

Pure GPU implementation of elasticity solver with:

- **Physics{Elasticity}** (renamed from Problem{Elasticity})
- **All boundary conditions in GPU kernels** (no CPU fallback)
- **Elements store geometry** via `update!(element, "geometry", nodes)`
- **Zero CPU-GPU transfer during solve** (matrix-free CG)
- **Breaking changes allowed** (old API deprecated for GPU performance)

## Design Principles

### 1. Everything on GPU

**Critical Rule:** All assembly, BC application, and solving happens on GPU device.

✅ **Allowed:**
```julia
# GPU kernels for assembly
@cuda threads=256 blocks=n compute_stresses_kernel!(...)

# BC application in device code
@cuda threads=256 blocks=n apply_dirichlet_kernel!(...)

# Matrix-free CG on GPU
K*u ≈ compute_residual_gpu!(u)
```

❌ **Not Allowed:**
```julia
# CPU loops
for element in elements
    # assemble on CPU
end

# CPU-GPU transfers in hot path
u_cpu = Array(u_gpu)  # SLOW!
process_on_cpu(u_cpu)
u_gpu = CuArray(u_cpu)
```

### 2. Physics{T} (Not Problem{T})

**Naming Convention:** Use "Physics" to emphasize field equations, not workflow.

```julia
# Field physics (volume elements)
physics = Physics(Elasticity, "body", 3)

# Boundary physics (surface elements) - future
bc = Physics(Dirichlet, "fixed", 3, "displacement")
```

**Why:** Clearer separation between:
- **Physics** = PDEs and constitutive laws
- **Solver** = Linear/nonlinear solution strategy
- **Analysis** = Time integration, load stepping

### 3. Elements Store Geometry

**No mesh object dependency.** Each element is self-contained.

```julia
el = Element(Tet4, [1, 2, 3, 4])

# Store coordinates
X = [0.0 1.0 0.0 0.0;
     0.0 0.0 1.0 0.0;
     0.0 0.0 0.0 1.0]
update!(el, "geometry", 0.0 => X)

# Store material
update!(el, "youngs modulus", 0.0 => 210e9)
update!(el, "poissons ratio", 0.0 => 0.3)

# Access during assembly
X_elem = el("geometry", time)
E = el("youngs modulus", time)
```

**Benefits:**
- Works with any mesh format (Gmsh, Abaqus, Code Aster, hand-coded)
- Heterogeneous materials (per-element properties)
- No global mesh data structure needed

### 4. BCs Integrated in Physics{Elasticity}

**Not separate problems.** BCs are part of the physics definition.

```julia
physics = Physics(Elasticity, "body", 3)

# Add body elements
add_elements!(physics, body_elements)

# Add Dirichlet BC (fixed displacement)
add_dirichlet!(physics, [1, 2, 3], [1,2,3], 0.0)  # Fix nodes 1,2,3 (all DOFs)

# Add Neumann BC (surface traction)
surf_el = Element(Tri3, [4, 5, 6])
update!(surf_el, "geometry", 0.0 => X_surf)
add_neumann!(physics, surf_el, Vec{3}((0.0, 0.0, -1e6)))  # Pressure

# Solve (BCs applied in GPU kernels)
result = solve_elasticity_gpu!(physics)
```

**GPU Implementation:**

**Dirichlet:**
```julia
# Flag array on GPU
is_fixed::CuArray{Bool,1}  # true if DOF is constrained

# Kernel applies BC
function apply_dirichlet_kernel!(r, is_fixed)
    dof = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if is_fixed[dof]
        r[dof] = 0.0
    end
end
```

**Neumann:**
```julia
# Surface element loop in GPU kernel
function apply_surface_traction_kernel!(
    f_ext, surface_nodes, surface_traction, nodes
)
    surf_idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    # Get surface element geometry
    n1, n2, n3 = surface_nodes[:, surf_idx]
    X1, X2, X3 = ...
    
    # Compute area and traction
    area = 0.5 * norm((X2-X1) × (X3-X1))
    force = area * traction / 3  # Lumped to nodes
    
    # Atomic add (allows parallel writes)
    CUDA.@atomic f_ext[3*n1-2] += force[1]
    ...
end
```

## File Structure

```
src/
├── gpu_physics_elasticity.jl    # NEW: Pure GPU Physics{Elasticity}
│   ├── Physics{Elasticity}      # Main type
│   ├── DirichletBC              # BC data structures
│   ├── NeumannBC
│   ├── GPUElasticityData        # GPU arrays
│   ├── CUDA kernels             # Device code
│   └── solve_elasticity_gpu!    # Main solver
│
├── assembly/problems.jl         # OLD: Problem{T} (CPU code)
├── problems_elasticity.jl       # OLD: CPU elasticity (to be renamed Physics)
└── JuliaFEM.jl                  # Main module (conditionally includes GPU)

demos/
└── cantilever_physics_gpu.jl    # NEW: Example usage

docs/src/book/design/
└── gpu_elasticity_refactoring.md  # Design document
```

## API Reference

### Types

```julia
# Physics type
mutable struct Physics{P}
    name::String
    dimension::Int
    properties::P                    # Elasticity instance
    body_elements::Vector{Element}
    bc_dirichlet::DirichletBC
    bc_neumann::NeumannBC
    gpu_data::Union{Nothing, GPUElasticityData}
end

# Elasticity properties
mutable struct Elasticity <: FieldProblem
    formulation::Symbol         # :continuum
    finite_strain::Bool
    geometric_stiffness::Bool
end
```

### Constructor

```julia
physics = Physics(Elasticity, name::String, dimension::Int)
```

**Example:**
```julia
physics = Physics(Elasticity, "cantilever beam", 3)
physics.properties.formulation = :continuum
physics.properties.finite_strain = false
```

### Adding Elements

```julia
add_elements!(physics::Physics{Elasticity}, elements::Vector{Element})
add_elements!(physics::Physics{Elasticity}, element::Element)
```

**Example:**
```julia
el = Element(Tet4, [1, 2, 3, 4])
update!(el, "geometry", 0.0 => X_elem)
update!(el, "youngs modulus", 0.0 => 210e9)
update!(el, "poissons ratio", 0.0 => 0.3)

add_elements!(physics, el)
```

### Adding Boundary Conditions

**Dirichlet (Fixed Displacement):**
```julia
add_dirichlet!(
    physics::Physics{Elasticity},
    node_ids::Vector{Int},
    components::Vector{Int},  # [1,2,3] for x,y,z
    value::Float64
)
```

**Example:**
```julia
# Fix all DOFs at nodes 1, 2, 3
add_dirichlet!(physics, [1, 2, 3], [1, 2, 3], 0.0)

# Prescribe Z displacement at node 10
add_dirichlet!(physics, [10], [3], -0.01)
```

**Neumann (Surface Traction):**
```julia
add_neumann!(
    physics::Physics{Elasticity},
    surface_element::Element,
    traction::Vec{3,Float64}
)
```

**Example:**
```julia
# Pressure load (1 MPa in -Z direction)
surf_el = Element(Tri3, [4, 5, 6])
update!(surf_el, "geometry", 0.0 => X_surf)
add_neumann!(physics, surf_el, Vec{3}((0.0, 0.0, -1e6)))
```

### Solving

```julia
result = solve_elasticity_gpu!(
    physics::Physics{Elasticity};
    time::Float64=0.0,
    tol=1e-6,
    max_iter=1000
)
```

**Returns:**
```julia
(
    u = u_cpu,           # Displacement vector (CPU array)
    iterations = iter,   # CG iterations
    residual = res       # Final residual norm
)
```

**Example:**
```julia
result = solve_elasticity_gpu!(physics; tol=1e-6, max_iter=500)

println("Converged in $(result.iterations) iterations")
println("Max displacement: $(maximum(abs.(result.u)))")
```

## Complete Example

```julia
using JuliaFEM
using CUDA
using Tensors

# 1. Create elements with geometry
nodes = [
    0.0  1.0  0.0  1.0
    0.0  0.0  1.0  1.0
    0.0  0.0  0.0  0.0
]

body_elements = Element[]
for conn in [[1,2,3,4]]
    el = Element(Tet4, conn)
    X_elem = nodes[:, conn]
    update!(el, "geometry", 0.0 => X_elem)
    update!(el, "youngs modulus", 0.0 => 210e9)
    update!(el, "poissons ratio", 0.0 => 0.3)
    push!(body_elements, el)
end

# 2. Create Physics{Elasticity}
physics = Physics(Elasticity, "body", 3)
add_elements!(physics, body_elements)

# 3. Add BCs
add_dirichlet!(physics, [1, 3], [1,2,3], 0.0)  # Fix nodes 1,3

surf = Element(Tri3, [2, 4])
update!(surf, "geometry", 0.0 => nodes[:, [2,4]])
add_neumann!(physics, surf, Vec{3}((0.0, 0.0, -1e6)))

# 4. Solve
result = solve_elasticity_gpu!(physics)

println("Max |u|: $(maximum(abs.(result.u)))")
```

## Migration from Old API

### Old (gpu_elasticity.jl - DEPRECATED)

```julia
# OLD API (Gmsh-dependent)
mesh = read_gmsh_mesh("beam.msh")
material = ElasticMaterial(210e9, 0.3)
fixed_nodes = get_surface_nodes(mesh, "FixedEnd")
pressure_nodes = get_surface_nodes(mesh, "PressureSurface")

physics = ElasticityPhysics(mesh, material, fixed_nodes, pressure_nodes, 1e6)
result = solve_elasticity_gpu(physics)
```

### New (gpu_physics_elasticity.jl - CURRENT)

```julia
# NEW API (mesh-agnostic, pure GPU)
# Create elements (from any source: Gmsh, Abaqus, hand-coded)
body_elements = [...]  # Elements with geometry

# Physics
physics = Physics(Elasticity, "body", 3)
add_elements!(physics, body_elements)

# BCs (integrated, not separate vectors)
add_dirichlet!(physics, fixed_nodes, [1,2,3], 0.0)

surf_elements = create_surface_elements(pressure_nodes, nodes)
for surf in surf_elements
    add_neumann!(physics, surf, Vec{3}((0.0, 0.0, -1e6)))
end

# Solve
result = solve_elasticity_gpu!(physics)
```

**Key Differences:**

1. **No GmshMesh** - Elements store geometry directly
2. **Physics{Elasticity}** instead of ElasticityPhysics struct
3. **BCs integrated** - Not separate node vectors
4. **Type-safe traction** - Vec{3,Float64} not scalar pressure
5. **Modular** - Can add multiple BC types

## GPU Kernel Architecture

### Phase 1: Compute Stresses at Integration Points

```julia
function compute_element_stresses_kernel!(σ_gp, u, nodes, elements, E, ν)
    # One thread per Gauss point
    # Computes stress from displacement gradient
    # Stores σ in global array
end
```

**Parallelism:** `n_elements × 4` threads (Tet4 has 4 Gauss points)

### Phase 2: Nodal Assembly (Internal Forces)

```julia
function nodal_assembly_kernel!(r, σ_gp, nodes, elements, node_to_elems)
    # One thread per node
    # Loops over touching elements
    # Accumulates f_int = ∫ B^T σ dV
end
```

**Parallelism:** `n_nodes` threads  
**Key:** No atomics needed (each node owned by one thread)

### Phase 3: Apply Surface Traction (Neumann BC)

```julia
function apply_surface_traction_kernel!(f_ext, surface_nodes, traction, nodes)
    # One thread per surface element
    # Computes area × traction
    # Atomic add to force vector
end
```

**Parallelism:** `n_surface_elements` threads  
**Atomics:** Required (multiple surfaces can share nodes)

### Phase 4: Apply Dirichlet BC

```julia
function apply_dirichlet_kernel!(r, is_fixed)
    # One thread per DOF
    # if is_fixed[dof]: r[dof] = 0.0
end
```

**Parallelism:** `3 × n_nodes` threads  
**Fast:** Just array indexing, no computation

### Phase 5: Conjugate Gradient Solver

```julia
function cg_solve_gpu!(gpu_data; tol, max_iter)
    while not converged
        # Matrix-free: Ap = K*p ≈ residual(p)
        Ap = compute_residual_gpu!(gpu_data, p)
        
        # CG update (all on GPU)
        alpha = dot(r, r) / dot(p, Ap)
        u .+= alpha .* p
        r .-= alpha .* Ap
        ...
    end
end
```

**Key:** All vectors stay on GPU (u, r, p, Ap)

## Performance Characteristics

**Strengths:**

✅ **Zero CPU-GPU transfer** during solve  
✅ **Matrix-free** (no memory for K matrix)  
✅ **Scalable** to millions of DOFs  
✅ **Type-stable** (all CuArrays, no Dicts)

**Limitations:**

⏳ **No preconditioning yet** (430 iterations for test case)  
⏳ **Tet4 only** (higher-order elements future work)  
⏳ **Linear elasticity only** (Newton-Krylov for nonlinear coming)

**Target Performance:**

- **Current:** 430 CG iterations for cantilever (190 nodes, 434 elements)
- **With Jacobi:** ~50-100 iterations
- **With ILU(0):** ~10-20 iterations (GPU ILU challenging)

## Future Work

### Phase 2: Preconditioning (Week 1-2)

**Goal:** Reduce CG iterations 430 → 10-20

**Approach:** Chebyshev-Jacobi preconditioner (GPU-friendly)

```julia
# Extract diagonal (matrix-free)
function extract_diagonal_kernel!(diag, nodes, elements, E, ν)
    # Compute K_ii by finite difference
    # Or: Assemble diagonal of B^T D B
end

# Apply preconditioner
M_inv = Diagonal(1 ./ diag)
CG(M_inv * K, M_inv * f)
```

### Phase 3: Nonlinear Elasticity (Week 3-6)

**Goal:** Newton-Krylov framework for plasticity

**Changes:**

1. **Material state at integration points**
   ```julia
   state_old::CuArray{PlasticState,1}  # Per Gauss point
   state_new::CuArray{PlasticState,1}
   ```

2. **Newton loop**
   ```julia
   while norm(R) > tol
       R = compute_residual_gpu!(u, state_old)
       K_tangent = approximate_jacobian_gpu(u, state_old)
       Δu = cg_solve_gpu!(K_tangent, -R)
       u += Δu
       update_state!(state_new, u)  # Trial state
   end
   commit_state!(state_old, state_new)  # Accept converged state
   ```

3. **Line search**
   ```julia
   α = backtracking_line_search_gpu(u, Δu, R)
   u += α * Δu
   ```

### Phase 4: Higher-Order Elements (Month 2-3)

**Goal:** Tri6, Tet10, Quad8, Hex20 support

**Approach:**

- Extend topology module (already has Tet10 definition)
- Add integration point data (more Gauss points)
- Generalize kernels (variable nodes per element)

### Phase 5: Contact Mechanics (Month 4-6)

**Goal:** Mortar contact on GPU

**Challenges:**

- Pairing algorithm (spatial search on GPU)
- Gap function evaluation (surface projections)
- Contact constraints (augmented Lagrangian)

## Testing

**Unit Tests:**
```julia
# Test BC application
@testset "Dirichlet BC" begin
    physics = Physics(Elasticity, "test", 3)
    add_dirichlet!(physics, [1], [1,2,3], 0.0)
    initialize_gpu!(physics)
    @test all(physics.gpu_data.is_fixed[[1,2,3]] .== true)
end
```

**Integration Tests:**
```julia
# Test cantilever beam
@testset "Cantilever GPU" begin
    physics = setup_cantilever()
    result = solve_elasticity_gpu!(physics)
    @test result.iterations < 500
    @test result.residual < 1e-6
end
```

**Validation Tests:**
```julia
# Compare to analytical solution
@testset "Beam bending" begin
    u_fem = solve_cantilever_gpu()
    u_analytical = beam_theory(E, I, L, P)
    @test isapprox(u_fem[end], u_analytical, rtol=0.1)
end
```

## References

- **Implementation:** `src/gpu_physics_elasticity.jl`
- **Demo:** `demos/cantilever_physics_gpu.jl`
- **Design:** `docs/src/book/design/gpu_elasticity_refactoring.md`
- **Old API:** `src/gpu_elasticity.jl` (DEPRECATED)

## Authors

- Jukka Aho (original author, maintainer)
- Refactored: November 10, 2025

## License

MIT (see LICENSE.md)
