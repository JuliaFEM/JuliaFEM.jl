# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
    AbstractPhysics

Abstract type for all physics problems in JuliaFEM.

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
- Concrete type: `Physics` in `src/physics/types.jl`
"""
abstract type AbstractPhysics end
