# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE

"""
Field System - Unified Multi-Field Architecture

# Core Principle: Multi-Field is Fundamental

**Single-field is just a special case of multi-field with one key!**

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

**Note**: `@NamedTuple` also works (since `DOFSet = NamedTuple`), but `@DOFSet` is preferred for future compatibility.

Each field is a `DOF{FieldType, EntityType}`:
- **FieldType**: `Displacement{Dim}`, `Temperature`, `DisplacementRotation{Dim}`, etc. (from AbstractField)
- **EntityType**: `Vertex`, `Edge`, `Face`, `Cell` (topological entity)

The field type encodes both the physical meaning and the underlying quantity type (Vec, Float64, etc.).

# Examples

**Single-field (displacement):**
```julia
S = @DOFSet{u::DOF{Displacement{3}, Vertex}}

Element{Tetrahedron{4}, Lagrange{1}, S}
```

**Multi-field (thermo-mechanical):**
```julia
S = @DOFSet{
    T::DOF{Temperature, Vertex},
    u::DOF{Displacement{3}, Vertex}
}

Element{Tetrahedron{4}, Lagrange{1}, S}
# DOF access: elem.dof_indices.T, elem.dof_indices.u
```

**Complex multi-physics (THM-E):**
```julia
S = @NamedTuple{
    T::DOF{Temperature, Vertex},           # Thermal
    u::DOF{Displacement{3}, Vertex},      # Mechanical
    p::DOF{Pressure, Cell},               # Hydraulic (discontinuous)
    φ::DOF{ElectricPotential, Edge}       # Electrical (Nédélec)
}
```

# Macro Sugar (@Fields was removed - use @DOFSet)

```julia
@DOFSet{
    T::DOF{Temperature, Vertex},
    u::DOF{Displacement{3}, Vertex},
    p::DOF{Pressure, Cell},
    φ::DOF{ElectricPotential, Edge}
}

# Equivalent to @NamedTuple:
@NamedTuple{
    T::DOF{Temperature, Vertex},
    u::DOF{Displacement{3}, Vertex},
    p::DOF{Pressure, Cell},
    φ::DOF{ElectricPotential, Edge}
}
```

# Compile-Time Computation

All field information is encoded in types:
```julia
# Number of DOFs for a field
@pure function field_ndofs(::Type{Tuple{T,E}}, ::Type{Topo}) where {T,E,Topo}
    return dof_size(T) * nentities(Topo, E)
end

# Total DOFs for element
@pure function ndofs(::Type{S}, ::Type{Topo}) where {S<:DOFSet, Topo}
    return sum(field -> field_ndofs(field, Topo), fieldtypes(S))
end
```

Everything resolves at compile time! Zero runtime overhead.

# Architecture Benefits

1. **Unified**: Single and multi-field use same code path
2. **Type-safe**: All dispatch on field specification type
3. **Named access**: `elem.dof_indices.T` self-documenting
4. **Extensible**: Add new fields without changing Element struct
5. **Efficient**: Compile-time computation, zero overhead
6. **Clean**: Multi-field is the general case, not a special case

# Migration Path

Old code with single DOF:
```julia
DOF{Vec{3}, Vertex}  # Old API
```

New unified field specification:
```julia
@NamedTuple{u::Tuple{Displacement{3}, Vertex}}  # New API (single field)
```

We can provide compatibility wrappers during migration.
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

The function extracts the field type from the tuple, then uses the `quantity_type()` trait
to get the underlying quantity type (Vec, Float64, etc.).

# See Also
- [`quantity_type(::Type{<:AbstractField})`](@ref) - Trait for field types
"""
# Support DOF{T,E} syntax (only syntax)
@inline function quantity_type(::Type{D}) where {D<:DOF}
    # Extract field type T from DOF{T,E}
    T = D.parameters[1]
    if T <: AbstractField
        return quantity_type(T)
    else
        error("Expected field type (AbstractField) in DOF, got $T. Use format: DOF{FieldType, EntityType}")
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
        error("Expected field type (AbstractField) in DOF, got $T. Use format: DOF{FieldType, EntityType}")
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
# Compatibility Helpers (for migration)
# ============================================================================

"""
    single_field(field_name::Symbol, quantity::Type, entity::Type)

Create a single-field specification.

# Example
```julia
S = single_field(:u, Vec{3}, Vertex)
# Equivalent to: @NamedTuple{u::Tuple{Vec{3}, Vertex}}
```
"""
function single_field(field_name::Symbol, quantity::Type, entity::Type)
    return @eval @NamedTuple{$field_name::Tuple{$quantity, $entity}}
end

"""
    field_names(::Type{S}) → NTuple{N,Symbol} where {S<:DOFSet}

Get field names from specification.

# Example
```julia
S = @NamedTuple{T::Tuple{Temperature,Vertex}, u::Tuple{Displacement{3},Vertex}}
field_names(S)  # (:T, :u)
```
"""
@inline field_names(::Type{S}) where {S<:DOFSet} = fieldnames(S)

"""
    field_count(::Type{S}) → Int where {S<:DOFSet}

Number of fields in specification.

# Example
```julia
S = @NamedTuple{T::Tuple{Temperature,Vertex}, u::Tuple{Displacement{3},Vertex}}
field_count(S)  # 2
```
"""
@inline field_count(::Type{S}) where {S<:DOFSet} = fieldcount(S)

"""
    is_single_field(::Type{S}) → Bool where {S<:DOFSet}

Check if specification has exactly one field.

# Example
```julia
S1 = @NamedTuple{u::Tuple{Displacement{3},Vertex}}
is_single_field(S1)  # true

S2 = @NamedTuple{T::Tuple{Temperature,Vertex}, u::Tuple{Displacement{3},Vertex}}
is_single_field(S2)  # false
```
"""
@inline is_single_field(::Type{S}) where {S<:DOFSet} = fieldcount(S) == 1

# ============================================================================
# Element Accessor Functions (convenience wrappers)
# ============================================================================

"""
    element_id(element) → UInt

Get element ID.

# Example
```julia
elem = Element{Triangle{3}, Lagrange{1}, S}(UInt(42), dof_indices)
element_id(elem)  # 42
```
"""
@inline element_id(element) = element.id

"""
    n_element_dofs(element) → Int

Total number of DOFs for this element (sum over all fields).

# Example
```julia
# Single field: u ∈ Vec{3} at 4 vertices → 12 DOFs
elem = Element{Tetrahedron{4}, Lagrange{1}, S_u}(id, (u=(1:12...,),))
n_element_dofs(elem)  # 12

# Multi-field: T + u → 4 + 12 = 16 DOFs
elem = Element{Tetrahedron{4}, Lagrange{1}, S_Tu}(id, (T=(1:4...,), u=(5:16...,)))
n_element_dofs(elem)  # 16
```
"""
@inline function n_element_dofs(element)
    total = 0
    for field_indices in element.dof_indices
        total += length(field_indices)
    end
    return total
end

"""
    element_dofs(element, field_name::Symbol) → Tuple{Int,...}

Get DOF indices for specific field.

# Example
```julia
S_THM = @DOFSet{
    T::Tuple{Temperature, Vertex},
    p::Tuple{Pressure, Vertex},
    u::Tuple{Displacement{3}, Vertex}
}
elem = Element{Tetrahedron{4}, Lagrange{1}, S_THM}(
    UInt(1),
    (T=(1,2,3,4), p=(5,6,7,8), u=(9,10,11,12,13,14,15,16,17,18,19,20))
)

element_dofs(elem, :T)  # (1, 2, 3, 4)
element_dofs(elem, :p)  # (5, 6, 7, 8)
element_dofs(elem, :u)  # (9, 10, ..., 20)
```
"""
@inline element_dofs(element, field_name::Symbol) = getfield(element.dof_indices, field_name)

"""
    basis_type(element) → Type{<:AbstractBasis}

Extract basis type from element type parameters.

# Example
```julia
elem = Element{Tetrahedron{4}, Lagrange{1}, S}(...)
basis_type(elem)  # Lagrange{1}
```
"""
@inline function basis_type(elem)
    T = typeof(elem)
    return T.parameters[2]  # Second type parameter is basis
end

"""
    dof_type(element) → Type{<:AbstractDOF}

Extract DOF specification type from element type parameters.

# Example
```julia
S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
elem = Element{Tetrahedron{4}, Lagrange{1}, S}(...)
dof_type(elem)  # Returns the S type (e.g., @NamedTuple{u::Tuple{Displacement{3},Vertex}})
```
"""
@inline function dof_type(elem)
    T = typeof(elem)
    return T.parameters[3]  # Third type parameter is DOF spec
end
