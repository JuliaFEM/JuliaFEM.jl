# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Physics API definitions.

This file defines physics problem abstractions - how we couple mesh, material,
field variables, and formulation into a solvable system.

Must be included after core api.jl, fields/api.jl, materials/api.jl, and formulations.
"""

# ============================================================================
# PHYSICS ABSTRACTIONS
# ============================================================================

"""
    AbstractPhysics

Abstract type for all physics problems.

Physics is the **coupling** of four components:
- **Mesh**: Topology/geometry (where)
- **Material**: Constitutive law (what material)
- **Field**: Solution variables (what we solve for)
- **Formulation**: Discretization strategy (how we discretize)

# Interface Requirements

All physics types must support:
- `assemble!(physics)` - Assemble global system matrices/operators
- `solve!(physics)` - Solve the physics problem
- `add_dirichlet!(physics, ...)` - Apply essential boundary conditions
- `add_neumann!(physics, ...)` - Apply natural boundary conditions

# Concrete Types

- `Physics{Formulation, Field, Mesh, Material}` - Standard FEM physics problem

# Design Philosophy

**Physics references Mesh (does not own it)**:
- Multiple physics can share one mesh (multiphysics coupling)
- No mesh duplication (memory efficient)
- Mesh is the topology owner
- Physics is the problem owner

**Type parameters enable dispatch specialization**:
```julia
# Specialize assembly for formulation × field combinations
assemble!(::Physics{ContinuumFormulation{FullThreeD}, Displacement{3}, M, Mat})
assemble!(::Physics{BeamFormulation{Timoshenko}, DisplacementRotation{3}, M, Mat})

# Generic fallback
assemble!(::Physics{Fm, F, M, Mat}) where {Fm, F, M, Mat}
```

# Examples

```julia
# 3D solid mechanics
physics = Physics(
    name = "cantilever",
    mesh = mesh,
    element_set = :all,
    field = Displacement{3}(),
    formulation = ContinuumFormulation{FullThreeD}(),
    material = LinearElastic(E=210e9, ν=0.3)
)

# Heat transfer
physics_thermal = Physics(
    name = "heat_conduction",
    mesh = mesh,
    element_set = :solid,
    field = Temperature(),
    formulation = ContinuumFormulation{FullThreeD}(),
    material = ThermalMaterial(k=50.0, ρ=7850.0, c=450.0)
)

# 3D beam
physics_beam = Physics(
    name = "beam",
    mesh = beam_mesh,
    element_set = :all,
    field = DisplacementRotation{3}(),
    formulation = BeamFormulation{Timoshenko}(),
    material = steel
)
```

# Multiphysics Example

```julia
# Share one mesh between structural and thermal physics
mesh = Mesh{Hex8}(nodes, connectivity)

physics_structural = Physics(
    name = "structure",
    mesh = mesh,  # Reference, not copy!
    field = Displacement{3}(),
    formulation = ContinuumFormulation{FullThreeD}(),
    material = steel
)

physics_thermal = Physics(
    name = "thermal",
    mesh = mesh,  # Same mesh reference!
    field = Temperature(),
    formulation = ContinuumFormulation{FullThreeD}(),
    material = thermal_steel
)
```

# See Also
- [`assemble!`](@ref) - System assembly
- [`solve!`](@ref) - Solve physics problem
- [`add_dirichlet!`](@ref) - Essential BCs
- [`add_neumann!`](@ref) - Natural BCs
- Concrete type: `Physics` in `src/physics.jl`
"""
abstract type AbstractPhysics end

# ============================================================================
# PHYSICS INTERFACE FUNCTIONS
# ============================================================================

"""
    assemble!(physics::AbstractPhysics) -> (K, f)

Assemble global system matrices for a physics problem.

This is the core FEM operation that builds:
- `K`: Global stiffness/tangent matrix (sparse or matrix-free operator)
- `f`: Global force/residual vector

# Arguments
- `physics`: Physics problem with mesh, material, field, formulation

# Returns
- `K`: Global stiffness matrix (sparse or operator)
- `f`: Global force vector

# Implementation Strategy

**Dispatch on formulation × field combination:**
```julia
# 3D solid mechanics with displacement field
function assemble!(physics::Physics{ContinuumFormulation{FullThreeD}, Displacement{3}, M, Mat})
    # Standard displacement-based elasticity
end

# Beam with displacement + rotation field
function assemble!(physics::Physics{BeamFormulation{Timoshenko}, DisplacementRotation{3}, M, Mat})
    # Beam-specific assembly (6 DOFs per node)
end

# Generic fallback
function assemble!(physics::Physics{Fm, F, M, Mat}) where {Fm, F, M, Mat}
    error("No assembly method for formulation $Fm with field $F")
end
```

# Examples

```julia
physics = Physics(
    name = "cantilever",
    mesh = mesh,
    element_set = :all,
    field = Displacement{3}(),
    formulation = ContinuumFormulation{FullThreeD}(),
    material = steel
)

# Assemble system
K, f = assemble!(physics)

# K is sparse matrix (for small problems) or matrix-free operator (for large)
# f is residual vector (nonlinear) or force vector (linear)
```

# See Also
- [`solve!`](@ref) - Solve after assembly
- Matrix-free assembly for large problems
- Nodal assembly for GPU acceleration
"""
function assemble! end

"""
    solve!(physics::AbstractPhysics) -> solution

Solve a physics problem.

Automatically selects appropriate solver based on problem type:
- Linear problems: Direct solver or CG/GMRES
- Nonlinear problems: Newton-Raphson with line search
- Dynamic problems: Time integration (Newmark, HHT-α)

# Arguments
- `physics`: Physics problem (must have BCs applied)

# Returns
- `solution`: Solution vector or solution object with field values

# Solver Selection Logic

```julia
function solve!(physics::Physics)
    if is_linear(physics.material)
        # Linear solver
        K, f = assemble!(physics)
        u = K \\ f
    else
        # Nonlinear solver (Newton-Raphson)
        u = newton_solve(physics)
    end
    return u
end
```

# Examples

```julia
# Apply boundary conditions first
add_dirichlet!(physics, fixed_nodes, [1,2,3], 0.0)
add_neumann!(physics, loaded_surface, traction)

# Solve
solution = solve!(physics)

# Access results
u = solution.displacement  # or solution.temperature, etc.
```

# See Also
- [`assemble!`](@ref) - System assembly
- [`add_dirichlet!`](@ref) - Essential BCs
- [`add_neumann!`](@ref) - Natural BCs
"""
function solve! end

"""
    add_dirichlet!(physics::AbstractPhysics, node_ids, components, values)

Apply essential (Dirichlet) boundary conditions.

Essential BCs prescribe field values at specific nodes:
- Fixed displacements (u = 0)
- Prescribed temperatures (T = 100°C)
- Prescribed rotations (θ = 0)

# Arguments
- `physics`: Physics problem
- `node_ids::Vector{Int}`: Node IDs where BC applies
- `components::Vector{Int}`: Which DOF components (e.g., [1,2,3] for all, [3] for z-direction)
- `values::Float64` or `Vector{Float64}`: Prescribed values

# Implementation Methods

**Elimination method** (modify K, f):
```julia
# Remove rows/columns for constrained DOFs
K_reduced, f_reduced = eliminate_constraints(K, f, bc_nodes, bc_values)
u_free = K_reduced \\ f_reduced
u_full = insert_constrained_values(u_free, bc_nodes, bc_values)
```

**Penalty method** (add large stiffness):
```julia
# Add penalty terms to K
for (node, component, value) in constraints
    dof = get_dof(node, component)
    K[dof, dof] += penalty  # Large number (e.g., 1e10 * max(K))
    f[dof] = penalty * value
end
```

**Lagrange multipliers** (augmented system):
```julia
# Solve [K  G^T] [u]   [f]
#       [G   0 ] [λ] = [g]
```

# Examples

```julia
# Fix all DOFs at nodes 1, 2, 3
add_dirichlet!(physics, [1,2,3], [1,2,3], 0.0)

# Fix only z-displacement at node 10
add_dirichlet!(physics, [10], [3], 0.0)

# Prescribe x-displacement at node 20
add_dirichlet!(physics, [20], [1], 0.1)

# Fix temperature at boundary nodes
add_dirichlet!(physics_thermal, boundary_nodes, [1], 273.15)
```

# See Also
- [`add_neumann!`](@ref) - Natural boundary conditions
- [`solve!`](@ref) - Solve with BCs applied
"""
function add_dirichlet! end

"""
    add_neumann!(physics::AbstractPhysics, surface_ids, values)

Apply natural (Neumann) boundary conditions.

Natural BCs apply forces, tractions, or fluxes:
- Surface tractions (solid mechanics)
- Heat flux (thermal)
- Pressure loads
- Body forces

# Arguments
- `physics`: Physics problem
- `surface_ids::Vector{Int}`: Surface/edge element IDs where BC applies
- `values`: Traction vectors (Vec{3}), scalars (pressure), or functions

# Implementation

Natural BCs contribute to force vector:
```julia
# For each surface element
for surf_elem in surface_elements
    # Integrate traction over surface
    f_surf = ∫(N^T * t * dS)  # N = shape functions, t = traction
    # Add to global force vector
    f[surf_dofs] += f_surf
end
```

# Examples

```julia
# Apply traction to surface
traction = Vec{3}(0.0, -1000.0, 0.0)  # 1000 N/m² downward
add_neumann!(physics, surface_elements, traction)

# Apply pressure (always normal to surface)
pressure = -100e3  # 100 kPa (negative = compression)
add_neumann!(physics, surface_elements, pressure)

# Heat flux (thermal problem)
heat_flux = 5000.0  # W/m²
add_neumann!(physics_thermal, surface_elements, heat_flux)
```

# See Also
- [`add_dirichlet!`](@ref) - Essential boundary conditions
- [`solve!`](@ref) - Solve with BCs applied
"""
function add_neumann! end
