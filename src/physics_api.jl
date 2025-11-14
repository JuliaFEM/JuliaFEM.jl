# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Backend-agnostic elasticity API

This file defines the user-facing API that works regardless of backend (CPU or GPU).
"""

# ============================================================================
# Abstract Types for Mesh, Material, Field, and Formulation
# ============================================================================

"""
    AbstractMesh

Abstract type for all mesh structures.

Concrete subtypes: Mesh{T<:AbstractTopology}
"""
abstract type AbstractMesh end

"""
    AbstractMaterial

Abstract type for all material models.

Concrete subtypes: LinearElastic, NeoHookean, PerfectPlasticity, etc.
"""
abstract type AbstractMaterial end

# ============================================================================
# Abstract Types for Field and Formulation (Double Dispatch)
# ============================================================================

"""
    AbstractField

Abstract type for field variables we solve for.

Subtypes define what physical quantity is being solved:
- `Displacement{Dim}`: Displacement field (3D, 2D, etc.)
- `Temperature`: Temperature field
- `DisplacementRotation{Dim}`: Displacement + rotation (beams, shells)
"""
abstract type AbstractField end

"""
    Displacement{Dim} <: AbstractField

Displacement field with Dim components per node.

# Examples
- `Displacement{3}()`: 3D displacement (ux, uy, uz)
- `Displacement{2}()`: 2D displacement (ux, uy)
"""
struct Displacement{Dim} <: AbstractField end

"""
    dofs_per_node(field::AbstractField) -> Int

Number of degrees of freedom per node for this field type.
"""
dofs_per_node(::Displacement{Dim}) where Dim = Dim

"""
    Temperature <: AbstractField

Temperature field (scalar per node).
"""
struct Temperature <: AbstractField end
dofs_per_node(::Temperature) = 1

"""
    DisplacementRotation{Dim} <: AbstractField

Combined displacement and rotation field (for beams, shells).

DOFs per node: 2*Dim (Dim displacements + Dim rotations)
- 3D: 6 DOFs (ux, uy, uz, θx, θy, θz)
- 2D: 4 DOFs (ux, uy, θz, warping)
"""
struct DisplacementRotation{Dim} <: AbstractField end
dofs_per_node(::DisplacementRotation{Dim}) where Dim = 2 * Dim

"""
    AbstractFormulation

Abstract type for discretization formulations.

Subtypes define HOW we discretize the governing equations:
- `ContinuumFormulation{Theory}`: Standard FEM for continuum mechanics
- `BeamFormulation{Theory}`: Beam elements (Euler-Bernoulli, Timoshenko)
- `ShellFormulation{Theory}`: Shell elements (Reissner-Mindlin, Kirchhoff)
- `TrussFormulation`: Truss elements
"""
abstract type AbstractFormulation end

"""
    AbstractContinuumTheory

Theory variants for continuum formulation.
"""
abstract type AbstractContinuumTheory end
struct FullThreeD <: AbstractContinuumTheory end
struct PlaneStress <: AbstractContinuumTheory end
struct PlaneStrain <: AbstractContinuumTheory end
struct Axisymmetric <: AbstractContinuumTheory end

"""
    ContinuumFormulation{Theory} <: AbstractFormulation

Standard continuum mechanics formulation with theory variant.

# Examples
- `ContinuumFormulation{FullThreeD}()`: Full 3D analysis
- `ContinuumFormulation{PlaneStress}()`: 2D plane stress
- `ContinuumFormulation{PlaneStrain}()`: 2D plane strain
"""
struct ContinuumFormulation{Theory<:AbstractContinuumTheory} <: AbstractFormulation end

"""
    AbstractBeamTheory

Theory variants for beam formulation.
"""
abstract type AbstractBeamTheory end
struct EulerBernoulli <: AbstractBeamTheory end
struct Timoshenko <: AbstractBeamTheory end

"""
    BeamFormulation{Theory} <: AbstractFormulation

Beam element formulation with theory variant.

# Examples
- `BeamFormulation{EulerBernoulli}()`: No shear deformation
- `BeamFormulation{Timoshenko}()`: Includes shear deformation
"""
struct BeamFormulation{Theory<:AbstractBeamTheory} <: AbstractFormulation end

"""
    TrussFormulation <: AbstractFormulation

Truss element formulation (axial force only).
"""
struct TrussFormulation <: AbstractFormulation end

# ============================================================================
# Physics Type Hierarchy
# ============================================================================

"""
    AbstractPhysics

Abstract base for all physics types.

JuliaFEM recognizes two fundamental categories:
1. Bulk Physics - operates on element sets (domain interiors)
2. Interface Physics - operates on surface pairs (domain boundaries)
"""
abstract type AbstractPhysics end

# ============================================================================
# Constraint Types (placeholder for future implementation)
# ============================================================================

"""
    Constraint

Placeholder for internal constraints (rigid body modes, incompressibility, etc.).
To be implemented in future stories.
"""
struct Constraint end

# ============================================================================
# Boundary Conditions
# ============================================================================

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

# ============================================================================
# Physics Struct (Fully Typed with Type Parameters)
# ============================================================================

"""
    Physics{Formulation<:AbstractFormulation, Field<:AbstractField, Mesh<:AbstractMesh, Material<:AbstractMaterial} <: AbstractPhysics

Bulk physics operating on element sets.

Represents volumetric phenomena:
- Solid mechanics (elasticity, plasticity, damage)
- Heat transfer
- Fluid dynamics
- Chemical diffusion

# Type Parameters (in dispatch priority order)
- `Formulation`: How we discretize - ContinuumFormulation{FullThreeD}, BeamFormulation{Timoshenko}, etc.
- `Field`: What we solve - Displacement{3}, Temperature, etc.
- `Mesh`: Mesh type - AbstractMesh subtype (e.g., Mesh, Mesh{Tet4})
- `Material`: Material type - AbstractMaterial subtype (e.g., LinearElastic, NeoHookean)

# Fields
- `name`: Problem name
- `mesh`: Reference to Mesh (topology owner - NOT copied!)
- `element_set`: Which elements in mesh this physics applies to
- `field`: Field instance (type Field)
- `formulation`: Formulation instance (type Formulation)
- `material`: Material properties (type Material)
- `constraints`: Optional internal constraints
- `bc_dirichlet`: Dirichlet boundary conditions
- `bc_neumann`: Neumann boundary conditions (surface loads)

# Type Parameter Order Rationale
**Formulation first** enables natural dispatch hierarchy:
```julia
# Most specific: Formulation + Field combination
assemble(::Physics{ContinuumFormulation{FullThreeD}, Displacement{3}, M, Mat}) = ...
assemble(::Physics{ContinuumFormulation{PlaneStress}, Displacement{2}, M, Mat}) = ...
assemble(::Physics{BeamFormulation{Timoshenko}, DisplacementRotation{3}, M, Mat}) = ...

# Generic fallbacks work naturally
assemble(::Physics{ContinuumFormulation, Temperature, M, Mat}) = ...  # Any continuum + thermal
assemble(::Physics{Fm, F, M, Mat}) where {Fm,F,M,Mat} = ...          # Fully generic
```

# Benefits of Type Parameters
- **Dispatch-optimized order:** Most important types first (formulation, then field)
- **Zero-cost dispatch:** Compiler selects assembly method at compile time
- **Type stability:** All field types known statically
- **Specialization:** Different algorithms per formulation × field combination
- **GPU-ready:** Concrete types enable device-specific optimization

# Design Philosophy
Physics REFERENCES Mesh (does not own it). Multiple Physics can share same Mesh.
Material properties stored in Physics, topology stored in Mesh.

# Example

```julia
mesh = Mesh("cantilever.inp")
material = LinearElastic(E=210e9, ν=0.3)

physics = Physics(
    name = "cantilever",
    mesh = mesh,
    element_set = :solid,
    field = Displacement{3}(),
    formulation = ContinuumFormulation{FullThreeD}(),
    material = material
)
# Type: Physics{ContinuumFormulation{FullThreeD}, Displacement{3}, typeof(mesh), LinearElastic}

add_dirichlet!(physics, [1,2,3], [1,2,3], 0.0)
sol = solve!(physics)
```

# Dispatch Example

```julia
# Different formulation/field combinations dispatch to specialized methods
physics_continuum_3d = Physics(..., formulation=ContinuumFormulation{FullThreeD}(), field=Displacement{3}())
physics_plane_stress = Physics(..., formulation=ContinuumFormulation{PlaneStress}(), field=Displacement{2}())
physics_beam = Physics(..., formulation=BeamFormulation{Timoshenko}(), field=DisplacementRotation{3}())
physics_thermal = Physics(..., formulation=ContinuumFormulation{FullThreeD}(), field=Temperature())

# Compiler generates specialized code for each formulation × field!
assemble(physics_continuum_3d)  # → assemble(::Physics{ContinuumFormulation{FullThreeD}, Displacement{3}, ...})
assemble(physics_plane_stress)  # → assemble(::Physics{ContinuumFormulation{PlaneStress}, Displacement{2}, ...})
assemble(physics_beam)          # → assemble(::Physics{BeamFormulation{Timoshenko}, DisplacementRotation{3}, ...})
assemble(physics_thermal)       # → assemble(::Physics{ContinuumFormulation{FullThreeD}, Temperature, ...})
```
"""
struct Physics{Formulation<:AbstractFormulation,Field<:AbstractField,Mesh<:AbstractMesh,Material<:AbstractMaterial} <: AbstractPhysics
    name::String
    mesh::Mesh                       # Fully typed mesh reference
    element_set::Symbol              # Which bulk elements
    field::Field                     # What we solve (Displacement{3}, Temperature, etc.)
    formulation::Formulation         # How we discretize (ContinuumFormulation{...}, etc.)
    material::Material               # Fully typed material
    constraints::Vector{Constraint}  # Optional: internal constraints
    bc_dirichlet::DirichletBC        # Essential boundary conditions
    bc_neumann::NeumannBC           # Natural boundary conditions

    # Inner constructor to ensure type parameters match field instances
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
        new{Formulation,Field,Mesh,Material}(name, mesh, element_set, field, formulation, material,
            constraints, bc_dirichlet, bc_neumann)
    end
end

"""
    Physics(; name, mesh, element_set, field, formulation, material, constraints=Constraint[])

Create a physics problem referencing a mesh.

Type parameters are automatically inferred from the argument types in dispatch-optimized order.

# Arguments (keyword arguments)
- `name::String`: Problem name
- `mesh`: Mesh reference (topology owner) - type becomes parameter Mesh
- `element_set::Symbol`: Which elements in mesh to use
- `field::AbstractField`: What we solve - type Field inferred
- `formulation::AbstractFormulation`: How we discretize - type Formulation inferred
- `material`: Material properties - type Material inferred
- `constraints`: Optional internal constraints (default: empty)

# Returns
`Physics{Formulation, Field, Mesh, Material}` where types are inferred from arguments

# Example

```julia
mesh = Mesh("model.inp")
material = LinearElastic(E=210e9, ν=0.3)

physics = Physics(
    name = "cantilever",
    mesh = mesh,
    element_set = :solid,
    field = Displacement{3}(),
    formulation = ContinuumFormulation{FullThreeD}(),
    material = material
)
# Type: Physics{ContinuumFormulation{FullThreeD}, Displacement{3}, typeof(mesh), LinearElastic}
```

# Type Inference

All type parameters are inferred automatically in dispatch priority order:
- `Formulation = typeof(formulation)` - formulation type (e.g., ContinuumFormulation{FullThreeD})
- `Field = typeof(field)` - field type (e.g., Displacement{3})
- `Mesh = typeof(mesh)` - concrete mesh type
- `Material = typeof(material)` - concrete material type

# Dispatch Benefits

Type parameter order optimized for natural dispatch:
```julia
# Most specific methods dispatch first
assemble(::Physics{ContinuumFormulation{FullThreeD}, Displacement{3}, M, Mat})
# Generic fallbacks work naturally
assemble(::Physics{Fm, F, M, Mat}) where {Fm, F, M, Mat}
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

    return Physics{Fm,F,M,Mat}(name, mesh, element_set, field, formulation, material,
        constraints, bc_dirichlet, bc_neumann)
end

# ============================================================================
# Deprecated API (Compatibility Layer)
# ============================================================================

# Note: Old Physics(::Type{P}, name, dimension) constructor is REMOVED
# The new type signature Physics{M,Mat,F,Fm} is incompatible with the old API
# Users must migrate to: Physics(name=..., mesh=..., field=..., formulation=..., material=...)

"""
    add_elements!(physics::Physics, elements::Vector{Element})

DEPRECATED: Physics no longer owns elements directly.

In the new API, Physics references Mesh. Elements are defined in Mesh,
and Physics selects which elements to use via `element_set`.

# Migration

Old API:
```julia
physics = Physics(Elasticity, "beam", 3)
add_elements!(physics, elements)
```

New API:
```julia
mesh = elements_to_mesh(elements)  # Convert elements to mesh
physics = Physics(
    name = "beam",
    mesh = mesh,
    element_set = :all,
    field = Displacement{3}(),
    formulation = ContinuumFormulation{FullThreeD}(),
    material = extract_material(elements[1])
)
```

This function will be removed in v2.0.
"""
function add_elements!(physics::Physics, elements::Vector{Element})
    @warn """
    add_elements!(physics, elements) is DEPRECATED.

    Physics now references Mesh instead of owning elements.
    Create a Mesh from elements and pass it to Physics constructor.

    Migration:
        mesh = elements_to_mesh(elements)
        physics = Physics(name=..., mesh=mesh, element_set=:all, ...)

    This function will be removed in v2.0.
    """ maxlog = 1

    # For backward compatibility, we need to somehow add elements to the mesh
    # This is hacky but maintains old API temporarily
    error("add_elements! no longer supported. Please migrate to new Physics(mesh=...) API.")
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
