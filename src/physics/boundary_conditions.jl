# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Boundary condition implementations for Physics.

This file provides concrete implementations of the boundary condition interface
functions defined in `src/physics/api.jl` for the `Physics` type.
"""

# ============================================================================
# BOUNDARY CONDITION METHODS (implement generic API)
# ============================================================================

"""
    add_dirichlet!(physics::Physics, node_ids::Vector{Int}, components::Vector{Int}, value::Float64)

Concrete implementation of Dirichlet BC application.

Applies essential (prescribed) boundary conditions to specified nodes and DOF components.

# Arguments
- `physics::Physics`: Physics problem
- `node_ids::Vector{Int}`: Node IDs where BC applies
- `components::Vector{Int}`: Which DOF components (e.g., [1,2,3] for all, [3] for z-direction)
- `value::Float64`: Prescribed value

# Examples

```julia
# Fix all DOFs at nodes 1, 2, 3
add_dirichlet!(physics, [1,2,3], [1,2,3], 0.0)

# Fix only z-displacement at node 10
add_dirichlet!(physics, [10], [3], 0.0)

# Prescribe x-displacement at node 20
add_dirichlet!(physics, [20], [1], 0.1)
```

See generic documentation in `src/physics/api.jl`.
"""
function add_dirichlet!(physics::Physics, node_ids::Vector{Int}, components::Vector{Int}, value::Float64)
    bc = physics.bc_dirichlet
    for node in node_ids
        push!(bc.node_ids, node)
        push!(bc.components, components)
        push!(bc.values, value)
    end
    return nothing
end

"""
    add_neumann!(physics::Physics, surface_ids::Vector{Int}, traction::Vec{3,Float64})

Concrete implementation of Neumann BC application.

Applies natural (force/traction) boundary conditions to specified surfaces.

# Arguments
- `physics::Physics`: Physics problem
- `surface_ids::Vector{Int}`: Surface/edge element IDs where BC applies
- `traction::Vec{3,Float64}`: Traction vector

# Examples

```julia
# Apply traction to surface
traction = Vec{3}(0.0, -1000.0, 0.0)  # 1000 N/m² downward
add_neumann!(physics, surface_elements, traction)

# Apply force to individual nodes (surface_ids are node IDs in this case)
force = Vec{3}(0.0, -100.0, 0.0)  # 100 N downward
add_neumann!(physics, loaded_nodes, force)
```

See generic documentation in `src/physics/api.jl`.
"""
function add_neumann!(physics::Physics, surface_ids::Vector{Int}, traction::Vec{3,Float64})
    bc = physics.bc_neumann
    for surf in surface_ids
        push!(bc.surface_ids, surf)
        push!(bc.values, traction)
    end
    return nothing
end
