# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# ============================================================================
# Compile-time helpers for ndofs / field_dof_range / local_dof_layout
# ============================================================================
# These are referenced from `@generated` functions defined later. In Julia
# 1.12+, generated bodies must see their helpers already bound at code-
# generation time, so the helpers live at the top of this file.

function _dof_per_entity(@nospecialize(Q))
    # `dof_size` is the single source of truth for the number of scalar
    # components per entity. New quantity types must add a `dof_size`
    # method (see `src/dofs/api.jl`); we deliberately do not swallow
    # errors here so that a missing method surfaces as a real
    # `MethodError` rather than a silent fall-through.
    return dof_size(Q)
end

function _count_entities_compiletime(@nospecialize(K), @nospecialize(E))
    # Must match runtime `count_entities(topology, entity_type)`.
    # `K` is a TYPE (e.g. `Tet4`), not an instance.
    if E === Vertex
        return nnodes(K)
    elseif E === Edge
        return nedges(K)
    elseif E === Face
        return nfaces(K)
    elseif E === Cell
        return 1
    else
        error("Unknown entity type $E")
    end
end

function _compile_time_ndofs(@nospecialize(field_type), @nospecialize(topology_type))
    # Field specs are `DOF{Quantity, Entity}`. The bare `Tuple{Q, E}`
    # form that older drafts used is no longer accepted by the
    # DOFHandler, so we don't support it here either.
    if field_type isa DataType && field_type <: DOF && length(field_type.parameters) == 2
        E = field_type.parameters[2]
        Q = quantity_type(field_type)
        return _dof_per_entity(Q) * _count_entities_compiletime(topology_type, E)
    else
        error("Cannot compute ndofs for field type $field_type (expected DOF{Quantity, Entity})")
    end
end

# ============================================================================
# Type-Level DOF Count Computation
# ============================================================================

"""
    ndofs(::Type{K}, ::Type{S}) → Int

Total number of DOFs for DOFSet `S` on topology `K`. Single source of truth:
delegates to `_compile_time_ndofs`, which is also used by
`field_dof_range` and `local_dof_layout`.

# Example
```julia
S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
ndofs(Tetrahedron{4}, S)  # → 12 (4 nodes × 3 components)

S2 = @DOFSet{u::DOF{Displacement{3}, Vertex}, p::DOF{Float64, Cell}}
ndofs(Tetrahedron{4}, S2) # → 13
```
"""
@generated function ndofs(::Type{K}, ::Type{S}) where {K, S}
    field_names = fieldnames(S)
    total = 0
    for fname in field_names
        field_spec = fieldtype(S, fname)
        total += _compile_time_ndofs(field_spec, K)
    end
    return total
end

"""
    AbstractElement{K, P, S, N}

Abstract supertype for finite elements following Ciarlet's triple (K, P, Σ).

# Type Parameters
- `K <: AbstractTopology`: Reference domain
- `P <: AbstractBasis`: Polynomial space
- `S`: Field specification (determines Σ functionals)
- `N::Int`: Total number of DOFs (inferred from S and K)

See `src/elements/README.md` for complete documentation.
"""
abstract type AbstractElement{K<:AbstractTopology, P<:AbstractBasis, S<:DOFSet, N} end

"""
    Element{K, P, S, N}

Finite element implementing Ciarlet's triple (K, P, Σ).

# Type Parameters
- `K`: Topology (Triangle{3}, Tetrahedron{4}, ...)
- `P`: Basis (Lagrange{1}, Lagrange{2}, ...)
- `S`: Field spec with quantity types and entity locations
- `N::Int`: Total DOF count (automatically inferred from S and K)

# Fields
- `id::UInt`: Element identifier
- `dof_indices::NTuple{N,UInt64}`: Flat tuple of global DOF indices

# Examples
```julia
# Single field: 3D displacement (12 DOFs = 4 nodes × 3 components)
S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
Element{Tetrahedron{4}, Lagrange{1}, S}(UInt(1), (1,2,3,4,5,6,7,8,9,10,11,12))

# Multi-field: Thermo-mechanical (16 DOFs = 4 T + 12 u)
S = @DOFSet{T::DOF{Temperature,Vertex}, u::DOF{Displacement{3},Vertex}}
Element{Tetrahedron{4}, Lagrange{1}, S}(UInt(1), (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16))
```

See `src/elements/README.md` for comprehensive documentation.
"""
struct Element{K<:AbstractTopology, P<:AbstractBasis, S<:DOFSet, N} <: AbstractElement{K,P,S,N}
    id::UInt
    dof_indices::NTuple{N,UInt64}
    
    # Inner constructor validates N matches spec
    function Element{K,P,S,N}(id::UInt, dof_indices::NTuple{N,UInt64}) where {K,P,S,N}
        expected = ndofs(K, S)
        if N != expected
            error("Element{$K,$P,$S,$N}: Expected $expected DOFs (from spec), got $N")
        end
        return new{K,P,S,N}(id, dof_indices)
    end
end

# Outer constructor infers N from tuple length
function Element{K,P,S}(id::UInt, dof_indices::NTuple{N,UInt64}) where {K,P,S,N}
    return Element{K,P,S,N}(id, dof_indices)
end

# Convenience constructor from varargs or vector
function Element{K,P,S}(id::UInt, dof_indices::UInt64...) where {K,P,S}
    return Element{K,P,S}(id, dof_indices)
end

function Element{K,P,S}(id::UInt, dof_indices::AbstractVector{<:Integer}) where {K,P,S}
    return Element{K,P,S}(id, tuple((UInt64(i) for i in dof_indices)...))
end

# ============================================================================
# Type-Level Queries
# ============================================================================

"""
    topology_type(::Element{K,P,S,N}) → Type{K}

Extract topology type K from element.
"""
topology_type(::Element{K,P,S,N}) where {K,P,S,N} = K
topology_type(::Type{Element{K,P,S,N}}) where {K,P,S,N} = K

"""
    basis_type(::Element{K,P,S,N}) → Type{P}

Extract basis type P from element.
"""
basis_type(::Element{K,P,S,N}) where {K,P,S,N} = P
basis_type(::Type{Element{K,P,S,N}}) where {K,P,S,N} = P

"""
    dof_type(::Element{K,P,S,N}) → Type{S}

Extract DOF specification type S from element.
"""
dof_type(::Element{K,P,S,N}) where {K,P,S,N} = S
dof_type(::Type{Element{K,P,S,N}}) where {K,P,S,N} = S

# ============================================================================
# Local-Global DOF Mapping for Coupled Assembly
# ============================================================================

"""
    local_dof_count(elem::Element) → Int

Total number of local DOFs for this element (sum over all fields).
"""
@inline function local_dof_count(elem::Element{K,P,S,N}) where {K,P,S,N}
    return N  # Now directly available as type parameter!
end

"""
    global_dof_indices(elem::Element) → Vector{UInt64}

Flattened vector of global DOF indices for this element.

See `src/elements/README.md` for assembly patterns.
"""
function global_dof_indices(elem::Element)
    return collect(elem.dof_indices)  # NTuple → Vector
end

"""
    local_to_global_map(elem::Element) → NTuple{N,UInt64}

Mapping from local DOF index to global DOF index.
`global_dof = map[local_dof]` where `local_dof ∈ 1:N`.

Returns tuple (not Vector) for type stability and compiler optimization.
Used for coupled assembly. See `src/elements/README.md`.
"""
@inline function local_to_global_map(elem::Element{K,P,S,N}) where {K,P,S,N}
    return elem.dof_indices  # Already flat!
end

# ============================================================================
# Local DOF Range Computation (COMPILE-TIME via @generated)
# ============================================================================
# Helpers `_compile_time_ndofs`, `_dof_per_entity`, `_count_entities_compiletime`
# are defined at the top of this file.

"""
    field_dof_range(elem::Element, field::Symbol) → UnitRange{Int}

Local DOF range for a specific field. Computed at compile time via @generated.
See `src/elements/README.md` for usage examples.
"""
@generated function field_dof_range(::Element{K,P,S,N}, field::Symbol) where {K,P,S,N}
    # This runs at COMPILE TIME!
    # S is the NamedTuple type containing field specifications
    
    if S <: NamedTuple
        # Multi-field case
        field_types = S.parameters[2]  # Tuple of field types
        field_names = fieldnames(S)
        
        # Compute offset for each field at compile time
        offset = 0
        field_ranges = Expr(:block)
        
        for (i, fname) in enumerate(field_names)
            ftype = field_types.parameters[i]
            n = _compile_time_ndofs(ftype, K)
            range_expr = :($offset+1:$offset+$n)
            
            # Generate: if field === :fname return range_expr end
            push!(field_ranges.args, quote
                if field === $(QuoteNode(fname))
                    return $range_expr
                end
            end)
            
            offset += n
        end
        
        # Add error case
        push!(field_ranges.args, :(error("Field ", field, " not found in element type $S")))
        
        return field_ranges
    else
        # Single-field case (S <: AbstractDOF)
        n = _compile_time_ndofs(S, K)
        return :(return 1:$n)
    end
end

# ============================================================================
# Element Queries
# ============================================================================

"""
    element_id(elem::Element) → UInt

Get element ID (index in mesh).
"""
element_id(elem::Element) = elem.id

"""
    element_dofs(elem::Element) → NTuple{N,UInt64}

Get all global DOF indices as flat tuple.
"""
element_dofs(elem::Element) = elem.dof_indices

"""
    element_dofs(elem::Element, field::Symbol) → Tuple

Get global DOF indices for specific field by extracting from flat tuple.

# Example
```julia
element_dofs(elem, :T)  # Extracts T indices from flat tuple
element_dofs(elem, :u)  # Extracts u indices from flat tuple
```
"""
function element_dofs(elem::Element{K,P,S,N}, field::Symbol) where {K,P,S,N}
    range = field_dof_range(elem, field)
    return elem.dof_indices[range]
end

"""
    n_element_dofs(elem::Element) → Int

Get total number of DOFs for this element (all fields).
"""
n_element_dofs(elem::Element{K,P,S,N}) where {K,P,S,N} = N

"""
    nnodes(::Element{K,P,S,N}) → Int

Get number of nodes from topology.
"""
nnodes(::Element{K,P,S,N}) where {K,P,S,N} = nnodes(K)
nnodes(::Type{Element{K,P,S,N}}) where {K,P,S,N} = nnodes(K)

# ============================================================================
# Compile-time DOF layout table (used by DOF-based assembler)
# ============================================================================

"""
    DOFLayoutEntry

Compile-time descriptor for one local DOF of an element. Used by the
DOF-based assembler to replace runtime `div`/`mod` decoding with pure
tuple lookups.

# Fields
- `field_idx::Int8`: index of the field inside the element's DOFSet (1-based)
- `entity_local::Int16`: local entity id within the element
  (1..`nnodes(K)` for `Vertex`, 1 for `Cell`, 1..`nedges(K)` for `Edge`,
  1..`nfaces(K)` for `Face`)
- `component::Int8`: component index inside the field's quantity
  (1 for a scalar, 1..3 for a Vec{3}, …)
"""
struct DOFLayoutEntry
    field_idx::Int8
    entity_local::Int16
    component::Int8
end

@inline field_idx(e::DOFLayoutEntry)    = Int(e.field_idx)
@inline entity_local(e::DOFLayoutEntry) = Int(e.entity_local)
@inline component(e::DOFLayoutEntry)    = Int(e.component)

"""
    local_dof_layout(::Type{Element{K,P,S,N}}) → NTuple{N, DOFLayoutEntry}

Compile-time DOF layout for an element template. The returned `NTuple`
has one entry per local DOF, in element-DOF order, describing which
field, which entity, and which component that DOF represents.

This is the central "Element-as-template" mechanism for the DOF-based
assembler: instead of decoding `local_i → (node, component)` with runtime
`div`/`mod`, the assembler indexes into this compile-time tuple, which
the compiler may unroll completely.

# Example
For `Element{Tet4, Lagrange{1}, @DOFSet{u::DOF{Vec{3}, Vertex}}, 12}`:
```
local_dof_layout(ET) ==
    (DOFLayoutEntry(1, 1, 1),  # u_x at vertex 1
     DOFLayoutEntry(1, 1, 2),  # u_y at vertex 1
     DOFLayoutEntry(1, 1, 3),  # u_z at vertex 1
     DOFLayoutEntry(1, 2, 1), …, DOFLayoutEntry(1, 4, 3))
```

For multi-field `(u::DOF{Vec{3},Vertex}, p::DOF{Float64,Cell})`:
```
local_dof_layout(ET) ==
    (DOFLayoutEntry(1, 1, 1), …, DOFLayoutEntry(1, N, 3),  # all u
     DOFLayoutEntry(2, 1, 1))                              # p
```
"""
@generated function local_dof_layout(::Type{Element{K, P, S, N}}) where {K, P, S, N}
    if !(S <: NamedTuple)
        return :(error("local_dof_layout: S=$($S) is not a DOFSet (NamedTuple)"))
    end

    field_names = fieldnames(S)
    entries = Expr[]

    for (fidx, fname) in enumerate(field_names)
        FT = fieldtype(S, fname)
        if !(FT <: DOF)
            return :(error("local_dof_layout: field :$($fname) of type $($FT) is not a DOF{Q,E}"))
        end
        Q_resolved = quantity_type(FT)
        E = FT.parameters[2]
        dpe = dof_size(Q_resolved)

        if E === Vertex
            n_entities = nnodes(K)
            for k in 1:n_entities, c in 1:dpe
                push!(entries, :(DOFLayoutEntry(Int8($fidx), Int16($k), Int8($c))))
            end
        elseif E === Cell
            for c in 1:dpe
                push!(entries, :(DOFLayoutEntry(Int8($fidx), Int16(1), Int8($c))))
            end
        elseif E === Edge
            n_ent = nedges(K)
            for k in 1:n_ent, c in 1:dpe
                push!(entries, :(DOFLayoutEntry(Int8($fidx), Int16($k), Int8($c))))
            end
        elseif E === Face
            n_ent = nfaces(K)
            for k in 1:n_ent, c in 1:dpe
                push!(entries, :(DOFLayoutEntry(Int8($fidx), Int16($k), Int8($c))))
            end
        else
            return :(error("local_dof_layout: entity type $($E) not yet supported"))
        end
    end

    if length(entries) != N
        return :(error("local_dof_layout: template Element{$($K),$($P),$($S),$($N)} expected " *
                       "$($N) DOFs, but layout yields $($(length(entries)))"))
    end

    return Expr(:tuple, entries...)
end

# Forwarding overload from instance
@inline local_dof_layout(::Element{K,P,S,N}) where {K,P,S,N} = local_dof_layout(Element{K,P,S,N})

# ============================================================================
# Display
# ============================================================================

function Base.show(io::IO, elem::Element{K,P,S,N}) where {K,P,S,N}
    print(io, "Element{$K, $P, $S}(id=$(elem.id), ndofs=$N)")
end
