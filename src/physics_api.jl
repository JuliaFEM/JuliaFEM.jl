# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Backend-agnostic elasticity API

This file defines the user-facing API that works regardless of backend (CPU or GPU).
"""

"""
    Elasticity <: FieldProblem

Elasticity physics problem type.

Properties:
- `formulation`: `:continuum` (3D solid) or `:plane_stress`, `:plane_strain`
- `finite_strain`: Enable geometric nonlinearity
- `geometric_stiffness`: Include stress-dependent stiffness
"""
mutable struct ElasticityPhysicsType <: FieldProblem
    formulation::Symbol         # :continuum (3D)
    finite_strain::Bool         # Geometric nonlinearity
    geometric_stiffness::Bool   # Stress-dependent stiffness
end

const Elasticity = ElasticityPhysicsType
ElasticityPhysicsType() = ElasticityPhysicsType(:continuum, false, false)

"""
    DirichletBC

Dirichlet boundary condition (prescribed displacement).

Fields:
- `node_ids`: Constrained node IDs
- `components`: Which components per node ([1,2,3] for all)
- `values`: Prescribed values per node
"""
struct DirichletBC
    node_ids::Vector{Int}           # Constrained nodes
    components::Vector{Vector{Int}} # Which components per node ([1,2,3] for all)
    values::Vector{Vector{Float64}} # Prescribed values per node
end

DirichletBC() = DirichletBC(Int[], Vector{Int}[], Vector{Float64}[])

"""
    NeumannBC

Neumann boundary condition (surface traction/pressure).

Fields:
- `surface_elements`: Surface elements (Tri3, Quad4, etc.) with geometry
- `traction`: Traction vector [t_x, t_y, t_z] per element [N/m²]
"""
struct NeumannBC
    surface_elements::Vector{Element}  # Surface elements with geometry
    traction::Vector{Vec{3,Float64}}   # Traction per element [N/m²]
end

NeumannBC() = NeumannBC(Element[], Vec{3,Float64}[])

"""
    Physics{P}

Container for physics problem with elements and boundary conditions.

Type parameter `P` is the physics type (e.g., Elasticity, HeatTransfer).

Fields:
- `name`: Problem name
- `dimension`: Spatial dimension (1, 2, or 3)
- `properties`: Physics-specific properties (e.g., Elasticity struct)
- `body_elements`: Volume elements with geometry and material
- `bc_dirichlet`: Dirichlet boundary conditions
- `bc_neumann`: Neumann boundary conditions (surface loads)

# Example

```julia
physics = Physics(Elasticity, "beam", 3)
add_elements!(physics, elements)
add_dirichlet!(physics, [1,2,3], [1,2,3], 0.0)
sol = solve!(physics)
```
"""
mutable struct Physics{P}
    name::String
    dimension::Int
    properties::P
    body_elements::Vector{Element}
    bc_dirichlet::DirichletBC
    bc_neumann::NeumannBC
end

"""
    Physics(::Type{P}, name::String, dimension::Int) where P

Create a physics problem of type P.

# Example

```julia
physics = Physics(Elasticity, "cantilever", 3)
```
"""
function Physics(::Type{P}, name::String, dimension::Int) where P
    properties = P()
    body_elements = Element[]
    bc_dirichlet = DirichletBC()
    bc_neumann = NeumannBC()
    return Physics{P}(name, dimension, properties, body_elements, bc_dirichlet, bc_neumann)
end

"""
    add_elements!(physics::Physics, elements::Vector{Element})

Add volume elements to the physics problem.

# Example

```julia
add_elements!(physics, [el1, el2, el3])
```
"""
function add_elements!(physics::Physics, elements::Vector{Element})
    append!(physics.body_elements, elements)
    return nothing
end

"""
    add_dirichlet!(physics::Physics, node_ids::Vector{Int}, components::Vector{Int}, value::Float64)

Add Dirichlet boundary condition (prescribed displacement).

# Arguments
- `node_ids`: Node IDs to constrain
- `components`: Which DOF components to constrain (1=x, 2=y, 3=z)
- `value`: Prescribed value

# Example

```julia
# Fix nodes 1,2,3 in all directions
add_dirichlet!(physics, [1,2,3], [1,2,3], 0.0)

# Fix nodes 4,5 in x-direction only
add_dirichlet!(physics, [4,5], [1], 0.0)
```
"""
function add_dirichlet!(physics::Physics, node_ids::Vector{Int}, components::Vector{Int}, value::Float64)
    # Expand: for each node, store which components are fixed
    for node in node_ids
        push!(physics.bc_dirichlet.node_ids, node)
        push!(physics.bc_dirichlet.components, components)
        push!(physics.bc_dirichlet.values, fill(value, length(components)))
    end
    return nothing
end

"""
    add_neumann!(physics::Physics, surface_element::Element, traction::Vec{3,Float64})

Add Neumann boundary condition (surface traction/pressure).

# Arguments
- `surface_element`: Surface element (Tri3, Quad4, etc.) with geometry
- `traction`: Traction vector [t_x, t_y, t_z] [N/m²]

# Example

```julia
# Apply pressure load
pressure = -1e6  # -1 MPa
traction = Vec{3}((0.0, 0.0, pressure))
add_neumann!(physics, surface_element, traction)
```
"""
function add_neumann!(physics::Physics, surface_element::Element, traction::Vec{3,Float64})
    push!(physics.bc_neumann.surface_elements, surface_element)
    push!(physics.bc_neumann.traction, traction)
    return nothing
end
