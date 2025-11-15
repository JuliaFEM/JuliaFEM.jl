# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Concrete Physics implementation.

This file implements the concrete `Physics` struct and its methods.
Abstract interface defined in `src/physics/api.jl`.

# Note
AbstractPhysics and interface functions (assemble!, solve!, add_dirichlet!, add_neumann!)
are now defined in src/physics/api.jl, which is included before this file in JuliaFEM.jl.
"""

# ============================================================================
# HELPER TYPES
# ============================================================================

"""
    Constraint

Internal constraint for constrained optimization (contact, incompressibility, etc.).

Placeholder for future constraint handling.
"""
struct Constraint end

# ============================================================================
# BOUNDARY CONDITION STORAGE
# ============================================================================

"""
    DirichletBC

Essential boundary conditions (prescribed displacements, temperatures, etc.).

# Fields
- `node_ids::Vector{Int}` - Node IDs with prescribed values
- `components::Vector{Vector{Int}}` - Which DOF components per node
- `values::Vector{Float64}` - Prescribed values
"""
mutable struct DirichletBC
    node_ids::Vector{Int}
    components::Vector{Vector{Int}}
    values::Vector{Float64}

    DirichletBC() = new(Int[], Vector{Int}[], Float64[])
end

"""
    NeumannBC

Natural boundary conditions (surface tractions, heat flux, etc.).

# Fields
- `surface_ids::Vector{Int}` - Surface/edge element IDs
- `values::Vector{Vec{3,Float64}}` - Traction vectors per surface
"""
mutable struct NeumannBC
    surface_ids::Vector{Int}
    values::Vector{Vec{3,Float64}}

    NeumannBC() = new(Int[], Vec{3,Float64}[])
end

# ============================================================================
# PHYSICS STRUCT
# ============================================================================

"""
    Physics{Formulation<:AbstractFormulation, Field<:AbstractField, Mesh<:AbstractMesh, Material<:AbstractMaterial} <: AbstractPhysics

Concrete physics implementation coupling mesh, material, field, and formulation.

# Type Parameters (dispatch-optimized order)
- `Formulation`: How we discretize (e.g., ContinuumFormulation{FullThreeD})
- `Field`: What we solve (e.g., Displacement{3}, Temperature)
- `Mesh`: Mesh type (e.g., Mesh{Hex8}, Mesh{Tet10})
- `Material`: Material type (e.g., LinearElastic, NeoHookean)

# Fields
- `name::String`: Problem name
- `mesh::Mesh`: Reference to mesh (topology owner - NOT copied!)
- `element_set::Symbol`: Which elements in mesh this physics applies to
- `field::Field`: Field instance
- `formulation::Formulation`: Formulation instance
- `material::Material`: Material properties
- `constraints::Vector{Constraint}`: Optional internal constraints
- `bc_dirichlet::DirichletBC`: Essential boundary conditions
- `bc_neumann::NeumannBC`: Natural boundary conditions

# Design Philosophy

**Physics references Mesh (does not own it)**. This enables:
- Multiple physics sharing one mesh (multiphysics)
- Memory efficiency (no mesh duplication)
- Natural domain decomposition

**Type parameter order** optimized for dispatch:
```julia
# Specialized methods for formulation × field combinations
assemble!(::Physics{ContinuumFormulation{FullThreeD}, Displacement{3}, M, Mat})
assemble!(::Physics{BeamFormulation{Timoshenko}, DisplacementRotation{3}, M, Mat})

# Generic fallback
assemble!(::Physics{Fm, F, M, Mat}) where {Fm,F,M,Mat}
```

# Example

```julia
mesh = Mesh{Hex8}(nodes, connectivity)
material = LinearElastic(E=210e9, ν=0.3)

physics = Physics(
    name = "cantilever",
    mesh = mesh,
    element_set = :all,
    field = Displacement{3}(),
    formulation = ContinuumFormulation{FullThreeD}(),
    material = material
)

add_dirichlet!(physics, [1,2,3], [1,2,3], 0.0)
sol = solve!(physics)
```
"""
struct Physics{Formulation<:AbstractFormulation,Field<:AbstractField,Mesh<:AbstractMesh,Material<:AbstractMaterial} <: AbstractPhysics
    name::String
    mesh::Mesh
    element_set::Symbol
    field::Field
    formulation::Formulation
    material::Material
    constraints::Vector{Constraint}
    bc_dirichlet::DirichletBC
    bc_neumann::NeumannBC

    # Inner constructor with type parameter validation
    function Physics{Formulation,Field,Mesh,Material}(
        name::String,
        mesh::Mesh,
        element_set::Symbol,
        field::Field,
        formulation::Formulation,
        material::Material,
        constraints::Vector{Constraint},
        bc_dirichlet::DirichletBC,
        bc_neumann::NeumannBC
    ) where {Formulation<:AbstractFormulation,Field<:AbstractField,Mesh<:AbstractMesh,Material<:AbstractMaterial}
        new{Formulation,Field,Mesh,Material}(
            name, mesh, element_set, field, formulation, material,
            constraints, bc_dirichlet, bc_neumann
        )
    end
end

# ============================================================================
# PHYSICS CONSTRUCTOR
# ============================================================================

"""
    Physics(; name, mesh, element_set, field, formulation, material, constraints=Constraint[])

Create a physics problem with automatic type inference.

# Keyword Arguments
- `name::String`: Problem name
- `mesh`: Mesh instance (topology owner)
- `element_set::Symbol`: Which elements in mesh to use (e.g., :all, :solid)
- `field`: Field instance (e.g., Displacement{3}(), Temperature())
- `formulation`: Formulation instance (e.g., ContinuumFormulation{FullThreeD}())
- `material`: Material instance (e.g., LinearElastic(E=210e9, ν=0.3))
- `constraints`: Optional constraints (default: empty)

# Returns
`Physics{Fm,F,M,Mat}` with fully inferred type parameters

# Example

```julia
physics = Physics(
    name = "cantilever",
    mesh = Mesh{Hex8}(nodes, connectivity),
    element_set = :all,
    field = Displacement{3}(),
    formulation = ContinuumFormulation{FullThreeD}(),
    material = LinearElastic(E=210e9, ν=0.3)
)
# Type: Physics{ContinuumFormulation{FullThreeD}, Displacement{3}, Mesh{Hex8}, LinearElastic}
```
"""
function Physics(;
    name::String,
    mesh::M,
    element_set::Symbol,
    field::F,
    formulation::Fm,
    material::Mat,
    constraints::Vector{Constraint}=Constraint[]
) where {M<:AbstractMesh,Mat<:AbstractMaterial,F<:AbstractField,Fm<:AbstractFormulation}
    bc_dirichlet = DirichletBC()
    bc_neumann = NeumannBC()

    return Physics{Fm,F,M,Mat}(
        name, mesh, element_set, field, formulation, material,
        constraints, bc_dirichlet, bc_neumann
    )
end

# ============================================================================
# BOUNDARY CONDITION METHODS (implement generic API)
# ============================================================================

"""
    add_dirichlet!(physics::Physics, node_ids::Vector{Int}, components::Vector{Int}, value::Float64)

Concrete implementation of Dirichlet BC application.

See generic documentation in `src/api.jl`.
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

See generic documentation in `src/api.jl`.
"""
function add_neumann!(physics::Physics, surface_ids::Vector{Int}, traction::Vec{3,Float64})
    bc = physics.bc_neumann
    for surf in surface_ids
        push!(bc.surface_ids, surf)
        push!(bc.values, traction)
    end
    return nothing
end

# ============================================================================
# ASSEMBLY AND SOLVER METHODS (stubs - full implementations elsewhere)
# ============================================================================

# assemble! and solve! implementations will be added in src/assembly/ and src/solvers/
# Those files will provide specialized methods dispatching on Physics type parameters
