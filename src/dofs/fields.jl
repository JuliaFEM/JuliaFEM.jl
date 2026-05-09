# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Field System - Unified Multi-Field Architecture

# Core Principle: Multi-Field is Fundamental

Single-field is just a special case of multi-field with one key!

Every element has a field specification `S` which is a `DOFSet` (currently implemented as NamedTuple):
```julia
S = @DOFSet{
    field_name::DOF{FieldType, EntityType}
}
```

# Field Specification Anatomy

```julia
@DOFSet{
    T::DOF{Temperature, Vertex},           # Temperature (scalar at vertices)
    u::DOF{Displacement{3}, Vertex},       # Displacement (vector at vertices)
    p::DOF{Pressure, Cell},                # Pressure (scalar at cells)
    φ::DOF{ElectricPotential, Edge}        # Electric (scalar on edges)
}
```

Note: `@NamedTuple` also works (since `DOFSet = NamedTuple`), but `@DOFSet` is preferred for future compatibility.

Each field is a `DOF{FieldType, EntityType}`:
- FieldType: `Displacement{Dim}`, `Temperature`, `DisplacementRotation{Dim}`, etc. (from AbstractField)
- EntityType: `Vertex`, `Edge`, `Face`, `Cell` (topological entity)

The field type encodes both the physical meaning and the underlying quantity type (Vec, Float64, etc.).

# Examples

Single-field (displacement):
```julia
S = @DOFSet{u::DOF{Displacement{3}, Vertex}}

Element{Tetrahedron{4}, Lagrange{1}, S}
```

Multi-field (thermo-mechanical):
```julia
S = @DOFSet{
    T::DOF{Temperature, Vertex},
    u::DOF{Displacement{3}, Vertex}
}

Element{Tetrahedron{4}, Lagrange{1}, S}
# DOF access (flat tuple, helper functions for per-field views):
#   element_dofs(elem)             # all global DOF indices, NTuple{N,UInt64}
#   element_dofs(elem, :T)         # global indices for field :T
#   field_dof_range(typeof(elem), :u)  # local index range for field :u
```

Complex multi-physics (THM-E):
```julia
S = @DOFSet{
    T::DOF{Temperature, Vertex},           # Thermal
    u::DOF{Displacement{3}, Vertex},       # Mechanical
    p::DOF{Pressure, Cell},                # Hydraulic (discontinuous)
    φ::DOF{ElectricPotential, Edge}        # Electrical (Nédélec)
}
```

# Macro Sugar

`@DOFSet{...}` is the preferred entry point. It currently expands to
`@NamedTuple{...}`, so the equivalent NamedTuple literal also works
provided every field type is a `DOF{Quantity, Entity}`. The bare
`Tuple{Quantity, Entity}` form that older drafts used is no longer
accepted by the DOFHandler.

# Compile-Time Computation

All field information is encoded in types:
```julia
# Number of DOFs for a field
Base.@pure function field_ndofs(::Type{D}, ::Type{Topo}) where {D<:DOF, Topo}
    return dof_size(quantity_type(D)) * nentities(Topo, entity_type(D))
end

# Total DOFs for element
Base.@pure function ndofs(::Type{S}, ::Type{Topo}) where {S<:DOFSet, Topo}
    return sum(field_ndofs(F, Topo) for F in fieldtypes(S))
end
```

Everything resolves at compile time! Zero runtime overhead.

# Architecture Benefits

1. Unified: Single and multi-field use same code path
2. Type-safe: All dispatch on field specification type
3. Named access: `element_dofs(elem, :T)` reads cleanly while
   `dof_indices` stays a flat `NTuple` for zero-allocation assembly
4. Extensible: Add new fields without changing Element struct
5. Efficient: Compile-time computation, zero overhead
6. Clean: Multi-field is the general case, not a special case
"""

# ============================================================================
# Field Query Functions
# ============================================================================

"""
    quantity_type(::Type{DOF{T,E}}) → Type

Extract quantity type from field specification.

Expects field types (AbstractField) in DOF:
- `DOF{Displacement{3}, Vertex}` → `Vec{3}` (via trait)
- `DOF{Temperature, Vertex}` → `Float64` (via trait)

# Examples
```julia
quantity_type(DOF{Displacement{3}, Vertex})   # Vec{3} (via trait)
quantity_type(DOF{Temperature, Vertex})       # Float64 (via trait)
```

# Implementation

The function extracts the field type `T` from `DOF{T,E}` and dispatches on
`quantity_type(::Type{T})` to get the underlying quantity type (Vec,
Float64, etc.).

# See Also
- [`quantity_type(::Type{<:AbstractField})`](@ref) - Trait for field types
"""
# Support DOF{T,E} syntax (only syntax)
@inline function quantity_type(::Type{D}) where {D<:DOF}
    # Extract field type T from DOF{T,E}
    T = D.parameters[1]
    if T <: AbstractField
        return quantity_type(T)
    elseif T === Float64 || T <: Vec || T <: Tensor || T <: SymmetricTensor
        # Already a raw quantity type — pass through.
        return T
    else
        error("Expected field type (AbstractField) or quantity type (Float64/Vec/Tensor) in DOF, got $T")
    end
end

"""
    entity_type(::Type{DOF{T,E}}) → Type{E}

Extract entity type from field specification.

# Examples
```julia
entity_type(DOF{Displacement{3}, Vertex})  # Returns Vertex
```
"""
# Support DOF{T,E} syntax (only syntax)
@inline entity_type(::Type{D}) where {D<:DOF} = D.parameters[2]

"""
    field_ndofs(::Type{DOF{T,E}}, ::Type{Topo}) → Int

Number of DOFs for a single field on given topology.

Computed at compile time: `dof_size(quantity_type(T)) × nentities(E, Topo)`

# Examples
```julia
field_ndofs(DOF{Displacement{3}, Vertex}, Tetrahedron{4})  # 3 × 4 = 12
```
"""
# Support DOF{T,E} syntax (only syntax)
@inline Base.@pure function field_ndofs(::Type{D}, ::Type{Topo}) where {D<:DOF, Topo}
    T = D.parameters[1]
    E = D.parameters[2]
    # T is a field type (AbstractField), need to get quantity type first
    if T <: AbstractField
        Q = quantity_type(T)  # Get underlying quantity type (Vec, Float64, etc.)
        return dof_size(Q) * nentities(Topo, E)
    else
        # Raw quantity (`DOF{Float64, Cell}`, `DOF{Float64, Face}`, …) —
        # same rule as single-field `ndofs(::DOF{T,E}, Topo)`.
        return dof_size(T) * nentities(Topo, E)
    end
end

# Single-field DOF{T,E} types
@inline Base.@pure function ndofs(::Type{D}, ::Type{Topo}) where {D<:DOF, Topo<:AbstractTopology}
    # Extract T and E from DOF{T,E}
    T = D.parameters[1]
    E = D.parameters[2]
    return dof_size(T) * nentities(Topo, E)  # Topology FIRST, entity SECOND
end

"""
    ndofs(::Type{S}, ::Type{Topo}) → Int where {S<:DOFSet}

Total DOFs for multi-field specification on given topology.

# Example
```julia
S = @DOFSet{T::DOF{Temperature,Vertex}, u::DOF{Displacement{3},Vertex}}
ndofs(S, Tetrahedron{4})  # 4 + 12 = 16
```
"""
@inline Base.@pure function ndofs(::Type{S}, ::Type{Topo}) where {S<:DOFSet, Topo<:AbstractTopology}
    # Sum ndofs for each field
    total = 0
    for name in fieldnames(S)
        FieldType = fieldtype(S, name)
        
        # Only support DOF{FieldType, Entity} format
        if FieldType <: DOF
            total += field_ndofs(FieldType, Topo)
        else
            error("Expected DOF{FieldType, Entity} for field :$name, got $FieldType. Use format: @DOFSet{field::DOF{FieldType, Entity}}")
        end
    end
    return total
end

# ============================================================================
# Field-Spec Accessors
# ============================================================================

"""
    field_names(::Type{S}) → NTuple{N,Symbol} where {S<:DOFSet}

Get field names from specification.

# Example
```julia
S = @DOFSet{T::DOF{Temperature, Vertex}, u::DOF{Displacement{3}, Vertex}}
field_names(S)  # (:T, :u)
```
"""
@inline field_names(::Type{S}) where {S<:DOFSet} = fieldnames(S)

"""
    field_count(::Type{S}) → Int where {S<:DOFSet}

Number of fields in specification.

# Example
```julia
S = @DOFSet{T::DOF{Temperature, Vertex}, u::DOF{Displacement{3}, Vertex}}
field_count(S)  # 2
```
"""
@inline field_count(::Type{S}) where {S<:DOFSet} = fieldcount(S)

"""
    is_single_field(::Type{S}) → Bool where {S<:DOFSet}

Check if specification has exactly one field.

# Example
```julia
S1 = @DOFSet{u::DOF{Displacement{3}, Vertex}}
is_single_field(S1)  # true

S2 = @DOFSet{T::DOF{Temperature, Vertex}, u::DOF{Displacement{3}, Vertex}}
is_single_field(S2)  # false
```
"""
@inline is_single_field(::Type{S}) where {S<:DOFSet} = fieldcount(S) == 1

# Element-level helpers (`element_id`, `element_dofs`, `n_element_dofs`,
# `basis_type`, `dof_type`) live in `src/elements/elements.jl`, where
# they are dispatched on `Element{K,P,S,N}`. Earlier generic versions
# that walked `element.dof_indices` as a NamedTuple were left over from
# a previous `Element` layout and have been removed.
