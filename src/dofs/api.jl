# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE

"""
DOF Type - Type-Level Field Specification

`DOF{T, E}` is an **abstract type** used purely for type-level field specifications.

# Philosophy

DOF types are NEVER instantiated - they exist purely at the type level to specify
field structure in a clean, readable way:

```julia
# Beautiful multi-field specification using DOFSet:
@DOFSet{
    u::DOF{Displacement{3}, Vertex},
    p::DOF{Pressure, Cell}
}
```

# Usage

**Multi-field specification (preferred):**
```julia
S = @NamedTuple{
    T::Tuple{Float64, Vertex},
    u::Tuple{Vec{3,Float64}, Vertex},
    p::Tuple{Float64, Cell}
}

# Or equivalently using DOF type syntax:
(T = DOF{Float64, Vertex}, u = DOF{Vec{3,Float64}, Vertex}, p = DOF{Float64, Cell})
```

**Single-field specification:**
```julia
S = @NamedTuple{u::Tuple{Vec{3,Float64}, Vertex}}

# Or using DOF type:
(u = DOF{Vec{3,Float64}, Vertex},)
```

# Connection to Assembly

The DOF specification `S` is a type parameter in Element:
```julia
Element{Tetrahedron{4}, Lagrange{1}, S}
```

The assembler uses `S` to:
1. Compute total DOF count: `ndofs(S, Tetrahedron{4})`
2. Extract field information: `fieldnames(S)`, `fieldtypes(S)`
3. Count DOFs per field: `field_ndofs(field_type, topology)`
4. Access element DOFs: `element.dof_indices.u`, `element.dof_indices.p`, etc.

All type-level, compile-time resolved! No runtime overhead.
"""

# ============================================================================
# DOF Size Function (utility for counting DOF components)
# ============================================================================

"""
    dof_size(::Type{T}) → Int

Number of DOF components for a quantity type.

# Examples
```julia
dof_size(Float64)             # 1 (scalar)
dof_size(Vec{2})              # 2 (2D vector)
dof_size(Vec{3})              # 3 (3D vector)
dof_size(Tensor{2,2})         # 4 (2×2 tensor)
dof_size(Tensor{2,3})         # 9 (3×3 tensor)
dof_size(SymmetricTensor{2,3}) # 6 (symmetric 3×3)
```
"""
Base.@pure dof_size(::Type{Float64}) = 1
Base.@pure dof_size(::Type{<:Vec{D}}) where {D} = D
Base.@pure dof_size(::Type{<:Tensor{2,D}}) where {D} = D * D
Base.@pure dof_size(::Type{<:SymmetricTensor{2,D}}) where {D} = div(D * (D + 1), 2)

# ============================================================================
# Abstract DOF Type (Base Type for All Field Specifications)
# ============================================================================

"""
    abstract type AbstractDOF

Abstract base type for all field specifications.

# Purpose
Base type for `DOF{T,E}` which is used inside `DOFSet` specifications:
```julia
S = @DOFSet{u::DOF{Displacement{3}, Vertex}}  # DOF{T,E} <: AbstractDOF
Element{K, P, S}  # where S <: DOFSet
```

# Subtypes
- `DOF{T,E}`: Type-level field specifications (never instantiated)

# Interface
Subtypes should support:
- `Base.fieldnames(::Type{<:AbstractDOF})` - field names as tuple
- `ndofs(::Type{<:AbstractDOF}, ::Type{<:AbstractTopology})` - total DOF count
"""
abstract type AbstractDOF end

# ============================================================================
# DOFSet - Multi-Field DOF Specification
# ============================================================================

"""
    DOFSet

Type alias for multi-field DOF specifications.

**Current implementation**: NamedTuple (but this is an implementation detail!)

Use `@DOFSet` macro to create multi-field specifications:
```julia
S = @DOFSet{T::DOF{Temperature, Vertex}, u::DOF{Displacement{3}, Vertex}}
```

# Purpose
Hides implementation detail so we can change from NamedTuple to custom struct later
without breaking user code.

# Usage in Element Type
```julia
# Single-field - always use @DOFSet
S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
Element{K, P, S}

# Multi-field - use @DOFSet
S = @DOFSet{T::DOF{Temperature, Vertex}, u::DOF{Displacement{3}, Vertex}}
Element{K, P, S}
```

# Compatibility Note
`NamedTuple` also works (since `DOFSet = NamedTuple`), but `DOFSet` is the preferred
interface. Using `DOFSet` ensures your code will continue to work if we change the
internal representation in the future.
"""
const DOFSet = NamedTuple

"""
    @DOFSet{field1::Type1, field2::Type2, ...}

Create a multi-field DOF specification.

# Syntax
Each field is specified using `DOF{FieldType, EntityType}`:
```julia
field_name::DOF{FieldType, EntityType}
```

Where:
- `FieldType`: AbstractField type (Displacement{3}, Temperature, etc.)
- `EntityType`: Topological entity (Vertex, Edge, Face, Cell)

# Examples
```julia
# Thermo-mechanical coupling
S = @DOFSet{T::DOF{Temperature, Vertex}, u::DOF{Displacement{3}, Vertex}}

# THM-E (full multi-physics)
S = @DOFSet{
    T::DOF{Temperature, Vertex},          # Temperature
    u::DOF{Displacement{3}, Vertex},      # Displacement
    p::DOF{Pressure, Cell},              # Pressure
    φ::DOF{ElectricPotential, Edge}      # Electric potential
}

# Create element with multi-field spec
element = Element{Tetrahedron{4}, Lagrange{1}, S}(id, dof_indices)
```

# Implementation Note
Currently wraps `@NamedTuple` but this is hidden from users.
Can be changed to custom struct later without breaking code!

# Compatibility
You can also use `@NamedTuple` directly, but `@DOFSet` is preferred for future compatibility.
"""
macro DOFSet(expr)
    return :(@NamedTuple($expr))
end

# ============================================================================
# DOF Type (Abstract - Type-Level Only)
# ============================================================================

"""
    abstract type DOF{T, E<:TopologicalEntity} <: AbstractDOF

Abstract type for type-level field specifications. NEVER instantiated!

Used purely for clean syntax when specifying fields:
```julia
(u = DOF{Vec{3,Float64}, Vertex}, p = DOF{Float64, Cell})
```

This is equivalent to the NamedTuple type:
```julia
@NamedTuple{u::Tuple{Vec{3,Float64}, Vertex}, p::Tuple{Float64, Cell}}
```

# Type Parameters
- `T`: Quantity type (Float64, Vec{D}, Tensor{2,D}, etc.)
- `E`: Topological entity (Vertex, Edge, Face, Cell)

# Examples
```julia
DOF{Float64, Vertex}              # Scalar at vertices
DOF{Vec{3,Float64}, Vertex}       # 3D vector at vertices
DOF{Tensor{2,3,Float64}, Cell}    # 3×3 tensor at cells
DOF{SymmetricTensor{2,2}, Edge}   # 2×2 symmetric tensor on edges
```
"""
abstract type DOF{T, E<:TopologicalEntity} <: AbstractDOF end

# DOF size for DOF type (type-level query)
Base.@pure function dof_size(::Type{<:DOF{T,E}}) where {T,E}
    return dof_size(T)
end

# Fieldnames for single-field DOF types (default field name is :dof)
Base.@pure Base.fieldnames(::Type{<:DOF}) = (:dof,)
