# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Field API definitions.

This file defines field variable abstractions - the physical quantities we solve for.
Must be included after core api.jl.
"""

# ============================================================================
# FIELD VARIABLE ABSTRACTIONS
# ============================================================================

"""
    AbstractField

Abstract type for field variables we solve for.

Field defines what physical quantity is being computed:
- Displacement (solid mechanics)
- Temperature (heat transfer)
- Velocity (fluid dynamics)
- DisplacementRotation (beams, shells)
- Pressure (hydrostatic problems)
- Electric potential (electrostatics)

# Interface Requirements

All field types must implement:
- `dofs_per_node(field)` - Number of DOFs per node

# Concrete Types
- `Displacement{Dim}` - Displacement field (Dim components)
- `Temperature` - Temperature field (1 component)
- `DisplacementRotation{Dim}` - Combined displacement + rotation (2*Dim components)

# Design Philosophy

Field type determines:
- DOF count per node
- Solution vector structure
- Boundary condition interpretation
- Assembly dispatch (different fields may need different assembly strategies)

# Examples

```julia
# 3D solid mechanics
field = Displacement{3}()
dofs_per_node(field)  # 3 (ux, uy, uz)

# Heat transfer
field = Temperature()
dofs_per_node(field)  # 1 (T)

# 3D beam/shell
field = DisplacementRotation{3}()
dofs_per_node(field)  # 6 (ux, uy, uz, θx, θy, θz)
```

# See Also
- [`dofs_per_node`](@ref)
"""
abstract type AbstractField end

"""
    Displacement{Dim} <: AbstractField

Displacement field with Dim components per node.

# Type Parameter
- `Dim`: Spatial dimension (1, 2, or 3)

# Examples
- `Displacement{3}()`: 3D displacement (ux, uy, uz)
- `Displacement{2}()`: 2D displacement (ux, uy)
- `Displacement{1}()`: 1D displacement (ux)

# DOFs per node
- 3D: 3 DOFs (ux, uy, uz)
- 2D: 2 DOFs (ux, uy)
- 1D: 1 DOF (ux)

# Usage

```julia
# 3D elasticity
physics = Physics(
    formulation=ContinuumFormulation{FullThreeD}(),
    field=Displacement{3}(),
    mesh=mesh,
    material=steel
)

# 2D plane stress
physics = Physics(
    formulation=ContinuumFormulation{PlaneStress}(),
    field=Displacement{2}(),
    mesh=mesh,
    material=aluminum
)
```
"""
struct Displacement{Dim} <: AbstractField end

"""
    Temperature <: AbstractField

Temperature field (scalar per node).

Used for heat transfer problems.

# DOFs per node
- 1 DOF (T)

# Usage

```julia
physics = Physics(
    formulation=ContinuumFormulation{FullThreeD}(),
    field=Temperature(),
    mesh=mesh,
    material=thermal_material
)
```

# See Also
- Heat transfer problems in docs/
"""
struct Temperature <: AbstractField end

"""
    DisplacementRotation{Dim} <: AbstractField

Combined displacement and rotation field (for beams, shells).

DOFs per node: 2*Dim (Dim displacements + Dim rotations)

# Type Parameter
- `Dim`: Spatial dimension (2 or 3)

# Examples
- `DisplacementRotation{3}()`: 3D beam/shell (ux, uy, uz, θx, θy, θz) - 6 DOFs
- `DisplacementRotation{2}()`: 2D beam (ux, uy, θz, warping) - 4 DOFs

# Usage

```julia
# 3D beam (Timoshenko theory)
physics = Physics(
    formulation=BeamFormulation{Timoshenko}(),
    field=DisplacementRotation{3}(),
    mesh=beam_mesh,
    material=steel
)

# 3D shell (Reissner-Mindlin theory)
physics = Physics(
    formulation=ShellFormulation{ReissnerMindlin}(),
    field=DisplacementRotation{3}(),
    mesh=shell_mesh,
    material=composite
)
```

# DOF Layout

3D (6 DOFs per node):
1. ux - displacement in x
2. uy - displacement in y  
3. uz - displacement in z
4. θx - rotation about x
5. θy - rotation about y
6. θz - rotation about z

2D (4 DOFs per node):
1. ux - displacement in x
2. uy - displacement in y
3. θz - rotation about z
4. ω - warping (for advanced beam theories)
"""
struct DisplacementRotation{Dim} <: AbstractField end

# ============================================================================
# FIELD INTERFACE FUNCTIONS
# ============================================================================

"""
    dofs_per_node(field::AbstractField) -> Int

Number of degrees of freedom per node for this field type.

# Examples

```julia
dofs_per_node(Displacement{3}())          # 3
dofs_per_node(Displacement{2}())          # 2
dofs_per_node(Temperature())              # 1
dofs_per_node(DisplacementRotation{3}())  # 6
dofs_per_node(DisplacementRotation{2}())  # 4
```

# Implementation

Field types must provide this method:
```julia
dofs_per_node(::Displacement{Dim}) where Dim = Dim
dofs_per_node(::Temperature) = 1
dofs_per_node(::DisplacementRotation{Dim}) where Dim = 2 * Dim
```
"""
function dofs_per_node end

# Concrete implementations
dofs_per_node(::Displacement{Dim}) where Dim = Dim
dofs_per_node(::Temperature) = 1
dofs_per_node(::DisplacementRotation{Dim}) where Dim = 2 * Dim

# ============================================================================
# QUANTITY TYPE TRAIT
# ============================================================================

"""
    quantity_type(::Type{<:AbstractField}) → Type

Extract underlying quantity type from field type.

The quantity type is the actual data type used to represent the field values:
- `Displacement{Dim}` → `Vec{Dim}` (vector quantity)
- `Temperature` → `Float64` (scalar quantity)
- `DisplacementRotation{Dim}` → `Vec{2*Dim}` (vector quantity with 2*Dim components)

# Examples

```julia
quantity_type(Displacement{3})          # Vec{3}
quantity_type(Displacement{2})          # Vec{2}
quantity_type(Temperature)              # Float64
quantity_type(DisplacementRotation{3})   # Vec{6}
quantity_type(DisplacementRotation{2})   # Vec{4}
```

# Implementation

Each field type must implement this trait. The quantity type is used for:
- DOF size computation
- Type inference in generated functions
- Field-to-physics mapping

# See Also
- [`dofs_per_node`](@ref) - Number of DOFs per node
"""
function quantity_type end

# Concrete implementations
quantity_type(::Type{Displacement{Dim}}) where Dim = Vec{Dim}
quantity_type(::Type{Temperature}) = Float64
quantity_type(::Type{DisplacementRotation{Dim}}) where Dim = Vec{2*Dim}

# ============================================================================
# DOF MAPPING - Maps nodes to global DOF indices
# ============================================================================

"""
    get_dof_mapping!(
        dofs::AbstractVector{Int},
        field::AbstractField,
        element_nodes::AbstractVector{Int}
    ) -> Nothing

Fill global DOF indices for an element **in-place** based on field type.

**Node-major ordering**: All DOFs for node 1, then node 2, etc.

# Arguments
- `dofs`: Pre-allocated DOF index buffer [ndofs_elem] (output)
- `field`: Field type (determines DOFs per node)
- `element_nodes`: Node indices for this element

# DOF Numbering Convention

For a field with `n` DOFs per node, node `k` has DOFs:
```
[n*(k-1)+1, n*(k-1)+2, ..., n*k]
```

# Examples

```julia
# 3D displacement (3 DOFs per node)
dofs = zeros(Int, 12)  # 4-node element × 3 DOFs
nodes = [10, 20, 30, 40]
get_dof_mapping!(dofs, Displacement{3}(), nodes)
# dofs = [28, 29, 30,  58, 59, 60,  88, 89, 90,  118, 119, 120]
#        |___________|  |___________|  |___________|  |____________|
#          node 10        node 20        node 30         node 40

# 2D displacement (2 DOFs per node)
dofs = zeros(Int, 8)  # 4-node element × 2 DOFs
get_dof_mapping!(dofs, Displacement{2}(), nodes)
# dofs = [19, 20,  39, 40,  59, 60,  79, 80]
#        |_____|   |_____|   |_____|   |_____|
#        node 10   node 20   node 30   node 40

# Temperature (1 DOF per node)
dofs = zeros(Int, 4)  # 4-node element × 1 DOF
get_dof_mapping!(dofs, Temperature(), nodes)
# dofs = [10, 20, 30, 40]
```

# Performance

Zero allocations - writes to pre-allocated buffer.

# See Also
- [`dofs_per_node`](@ref) - Number of DOFs per node
"""
@inline function get_dof_mapping!(
    dofs::AbstractVector{Int},
    field::AbstractField,
    element_nodes::AbstractVector{Int}
)
    ndofs_per_node = dofs_per_node(field)

    # Use indexed loop to avoid iterator allocation
    nnodes = length(element_nodes)
    idx = 1
    @inbounds for i in 1:nnodes
        node_id = element_nodes[i]
        for component in 1:ndofs_per_node
            dofs[idx] = ndofs_per_node * (node_id - 1) + component
            idx += 1
        end
    end

    return nothing
end
