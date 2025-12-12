# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/FEMBase.jl/blob/master/LICENSE

# ============================================================================
# Type-Level DOF Count Computation
# ============================================================================

"""
    ndofs(::Type{K}, ::Type{S}) → Int

Compute total number of DOFs for field spec S on topology K.

# Example
```julia
S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
ndofs(Tetrahedron{4}, S)  # → 12 (4 nodes × 3 components)
```
"""
@generated function ndofs(::Type{K}, ::Type{S}) where {K, S}
    field_names = fieldnames(S)
    total = 0
    
    for fname in field_names
        field_spec = fieldtype(S, fname)  # Tuple{Displacement{3}, Vertex}
        field_type = field_spec.parameters[1]  # Displacement{3}
        entity_type = field_spec.parameters[2]  # Vertex
        
        # Extract quantity type via trait
        Q = quantity_type(field_spec)  # Vec{3}
        
        # Count entities
        n_entities = if entity_type === Vertex
            nnodes(K())
        elseif entity_type === Edge
            nedges(K())
        elseif entity_type === Face
            nfaces(K())
        else
            error("Unsupported entity type: $entity_type")
        end
        
        # Count components per entity
        n_components = if Q === Float64
            1
        elseif Q isa UnionAll && Q.body <: Tensor && Q.body.parameters[1] == 1
            Q.body.parameters[2]
        else
            error("Unsupported quantity type: $Q")
        end
        
        total += n_entities * n_components
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
# Compile-Time Helper Functions for @generated field_dof_range
# ============================================================================

# Helper: Compute ndofs at compile time
function _compile_time_ndofs(@nospecialize(field_type), @nospecialize(topology_type))
    # Handle DOF{FieldType, EntityType} format (new format)
    if field_type isa DataType && field_type <: DOF && length(field_type.parameters) == 2
        FieldType = field_type.parameters[1]  # e.g., Displacement{3}
        E = field_type.parameters[2]          # e.g., Vertex
        
        # Extract quantity type via trait (handles Displacement{3} → Vec{3})
        Q = quantity_type(field_type)
        
        # Number of DOFs = dof_per_entity * number_of_entities
        return _dof_per_entity(Q) * _count_entities_compiletime(topology_type, E)
    # Handle Tuple{FieldType, E} format (legacy format)
    elseif field_type isa DataType && field_type <: Tuple && length(field_type.parameters) == 2
        FieldType = field_type.parameters[1]  # Could be Displacement{3} or Vec{3}
        E = field_type.parameters[2]
        
        # Extract quantity type via trait (handles both field types and quantity types)
        Q = quantity_type(field_type)
        
        # Number of DOFs = dof_per_entity * number_of_entities
        return _dof_per_entity(Q) * _count_entities_compiletime(topology_type, E)
    else
        error("Cannot compute ndofs for field type $field_type (expected DOF{...} or Tuple{...})")
    end
end

function _dof_per_entity(@nospecialize(Q))
    # Use dof_size which handles all quantity types properly (Displacement{3}, Vec{3}, Float64, UnionAll, etc.)
    # This is the most robust approach
    try
        return dof_size(Q)
    catch e
        # Fallback for specific cases if dof_size fails
        if Q === Float64
            return 1
        else
            error("Cannot determine dof_size for quantity type $Q: $e")
        end
    end
end

function _count_entities_compiletime(@nospecialize(K), @nospecialize(E))
    # This must match count_entities(topology, entity_type) at runtime
    # K is a TYPE (e.g., Tet4), not an instance
    if E === Vertex
        return nnodes(K)  # nnodes accepts Type
    elseif E === Edge
        return nedges(K)  # nedges accepts Type
    elseif E === Face
        return nfaces(K)  # nfaces accepts Type
    elseif E === Cell
        return 1  # One cell per element
    else
        error("Unknown entity type $E")
    end
end

# ============================================================================
# Local DOF Range Computation (COMPILE-TIME via @generated)
# ============================================================================

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
# Local-to-Global Mapping (Type-Stable Tuple Version)
# ============================================================================

# ============================================================================
# Type Extraction (previously defined above)
# ============================================================================

# These were defined earlier but are here for reference
# topology_type, basis_type, dof_type already defined above

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
# Display
# ============================================================================

function Base.show(io::IO, elem::Element{K,P,S,N}) where {K,P,S,N}
    print(io, "Element{$K, $P, $S}(id=$(elem.id), ndofs=$N)")
end
