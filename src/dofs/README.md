# DOF System

Type-level field specifications for finite elements.

---

## Philosophy

**Multi-field is fundamental. Single-field is just a special case with one key.**

Field specifications are `DOFSet` types (currently implemented as NamedTuples), used purely at the type level:

```julia
# Clean syntax using abstract DOF{T,E} types:
(T = DOF{Float64, Vertex}, u = DOF{Vec{3,Float64}, Vertex})

# Using DOFSet macro (preferred):
S = @DOFSet{T::DOF{Temperature, Vertex}, u::DOF{Displacement{3}, Vertex}}

# Used in Element type parameter:
Element{Tetrahedron{4}, Lagrange{1}, S}
```

**Note**: `DOFSet` is currently an alias for `NamedTuple`, but using `DOFSet` ensures your code will continue to work if we change the internal representation in the future. You can also use `@NamedTuple` directly, but `@DOFSet` is the preferred interface.

---

## Core Concept: DOF{T,E}

`DOF{T, E}` is an **abstract type** for type-level specifications (never instantiated):

```julia
abstract type DOF{T, E<:TopologicalEntity} end
```

**Type Parameters:**

- `T`: Quantity type (Float64, Vec{D}, Tensor{2,D}, etc.)
- `E`: Topological entity (Vertex, Edge, Face, Cell)

**Examples:**

```julia
DOF{Float64, Vertex}              # Scalar at vertices
DOF{Vec{3,Float64}, Vertex}       # 3D vector at vertices
DOF{Tensor{2,3,Float64}, Cell}    # 3×3 tensor at cells
```

---

## Field Specifications with @DOFSet

Use the `@DOFSet` macro for multi-field specifications:

```julia
# Multi-field specification with DOF syntax (preferred)
S = @DOFSet{
    T::DOF{Temperature, Vertex},
    u::DOF{Displacement{3}, Vertex},
    p::DOF{Pressure, Cell}
}

# Returns: DOFSet type (currently NamedTuple, but this is an implementation detail)
```

**Always use `@DOFSet`, even for single-field elements:**

```julia
# Single-field - always wrap in @DOFSet
S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
Element{Tetrahedron{4}, Lagrange{1}, S}
```

This simplifies generated code by having a single, consistent syntax for all cases.

---

## Element Integration

Field specification `S` is the third type parameter in Element:

```julia
struct Element{K<:AbstractTopology, P<:AbstractBasis, S<:DOFSet, N}
    id::UInt
    dof_indices::NTuple{N,UInt64}  # Flat tuple of global DOF indices
end
```

**Type Parameters:**
- `K`: Topology (Triangle{3}, Tetrahedron{4}, ...)
- `P`: Basis (Lagrange{1}, Lagrange{2}, ...)
- `S`: Field specification (`DOF{T,E}` for single-field, `DOFSet` for multi-field)
- `N::Int`: Total DOF count (automatically inferred from S and K)

**Example with single field:**

```julia
# Most common case - use DOF{T,E} directly
S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
elem = Element{Tetrahedron{4}, Lagrange{1}, S}(
    UInt(1),
    (dof=(1,2,3,4,5,6,7,8,9,10,11,12),)  # Single field, default name :dof
)
```

**Example with multiple fields:**

```julia
S = @DOFSet{
    T::DOF{Temperature, Vertex},
    u::DOF{Displacement{3}, Vertex}
}

elem = Element{Tetrahedron{4}, Lagrange{1}, S}(
    UInt(1),
    (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16)  # Flat tuple: T DOFs first, then u DOFs
)
```

---

## DOF Counting

All DOF information is computed at compile time:

```julia
# Components per quantity type:
dof_size(Float64) = 1
dof_size(Vec{3}) = 3
dof_size(Tensor{2,3}) = 9

# DOFs per field:
field_ndofs(DOF{Temperature, Vertex}, Tetrahedron{4})  # 1 × 4 = 4
field_ndofs(DOF{Displacement{3}, Vertex}, Tetrahedron{4})   # 3 × 4 = 12

# Total DOFs for element:
ndofs(S, Tetrahedron{4})  # Sum over all fields
```

---

## Examples

### Single-Field Elements

```julia
# Heat conduction:
S = @Fields begin
    T: Float64, Vertex
end
Element{Triangle{3}, Lagrange{1}, S}  # 3 DOFs

# 3D Elasticity:
S = @Fields begin
    u: Vec{3,Float64}, Vertex
end
Element{Tetrahedron{4}, Lagrange{1}, S}  # 12 DOFs
```

### Multi-Field Elements

```julia
# Thermo-mechanical:
S = @Fields begin
    T: Float64, Vertex
    u: Vec{3,Float64}, Vertex
end
Element{Tetrahedron{4}, Lagrange{1}, S}  # 4 + 12 = 16 DOFs

# Thermo-hydro-mechanical:
S = @Fields begin
    T: Float64, Vertex     # Temperature at vertices
    p: Float64, Vertex     # Pressure at vertices
    u: Vec{3,Float64}, Vertex   # Displacement at vertices
end
Element{Tetrahedron{4}, Lagrange{1}, S}  # 4 + 4 + 12 = 20 DOFs
```

---

## Key Benefits

1. **Type-level computation** - All DOF information resolved at compile time
2. **Named field access** - `elem.dof_indices.T` self-documenting
3. **Multi-field natural** - No special cases, just more keys in NamedTuple
4. **Zero overhead** - No runtime type checks, no Dict lookups
5. **Extensible** - Add new fields without changing Element struct

---

## Module Structure

- **`api.jl`** - DOF{T,E} abstract type + dof_size() utility
- **`fields.jl`** - FieldSpec, @Fields macro, ndofs(), field queries
- **`dof_manager.jl`** - Global DOF numbering and element creation
- **`dofs.jl`** - Main entry point

---

## Related Modules

- `/src/topology/` - TopologicalEntity types (Vertex, Edge, Face, Cell)
- `/src/basis/` - Basis functions (Lagrange, etc.)
- `/src/elements/` - Element implementation
