# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Field information for DOF-based assembly.

Provides decoding of local DOF indices to field, entity, and component information.
"""

using ..JuliaFEM: AbstractElement, field_dof_range, topology_type, nnodes, Vertex

"""
    DOFFieldInfo

Information about a local DOF within an element.

# Fields
- `field::Symbol`: Field name (:u, :T, :p, etc.)
- `entity_type::Type`: Topological entity (Vertex, Edge, Face, Cell)
- `entity_idx::Int`: Index of entity (1-based, e.g., node 1, node 2, ...)
- `component::Int`: Component index (1 for scalar, 1-3 for Vec{3}, etc.)
- `node_idx::Int`: Node index (for Vertex entities, same as entity_idx)
"""
struct DOFFieldInfo
    field::Symbol
    entity_type::Type
    entity_idx::Int
    component::Int
    node_idx::Int  # Convenience: for Vertex entities, same as entity_idx
end

"""
    decode_local_dof(element::Element, local_dof_idx::Int) -> DOFFieldInfo

Decode which field, entity, and component a local DOF index represents.

**Single-field case**: For `@DOFSet{u::DOF{Displacement{3}, Vertex}}`, 
local DOF indices are ordered as:
- DOF 1,2,3 = u at node 1 (ux, uy, uz)
- DOF 4,5,6 = u at node 2 (ux, uy, uz)
- etc.

**Multi-field case**: Uses `field_dof_range()` to determine which field,
then decodes within that field.

# Arguments
- `element`: Element with DOF specification
- `local_dof_idx`: Local DOF index (1-based)

# Returns
- `DOFFieldInfo` with field, entity, component information

# Example

```julia
# Single-field: 3D displacement
S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
elem = Element{Tet4, Lagrange{1}, S}(...)
info = decode_local_dof(elem, 1)  # DOFFieldInfo(:u, Vertex, 1, 1) - ux at node 1
info = decode_local_dof(elem, 2)  # DOFFieldInfo(:u, Vertex, 1, 2) - uy at node 1
info = decode_local_dof(elem, 4)  # DOFFieldInfo(:u, Vertex, 2, 1) - ux at node 2
```
"""
function decode_local_dof(element::AbstractElement, local_dof_idx::Int)
    S = typeof(element).parameters[3]  # DOF specification type (NamedTuple for DOFSet)
    
    # S is always a NamedTuple (DOFSet), even for single-field
    # Single-field: @NamedTuple{dof::DOF{...}}
    # Multi-field: @NamedTuple{field1::DOF{...}, field2::DOF{...}, ...}
    if S <: NamedTuple
        # Multi-field case: use field_dof_range to find which field
        field_names = fieldnames(S)
        
        dof_offset = 0
        for field_name in field_names
            range = field_dof_range(element, field_name)
            range_start = first(range)
            range_end = last(range)
            
            if local_dof_idx >= range_start && local_dof_idx <= range_end
                # Found the field! Now decode within this field
                dof_in_field = local_dof_idx - range_start + 1
                
                # Get field type to determine entity and component
                field_type = fieldtype(S, field_name)
                
                # Extract entity type and quantity type
                # Field type is DOF{QuantityType, EntityType}
                if field_type <: DOF
                    QuantityType = field_type.parameters[1]  # e.g., Displacement{3} or Vec{3}
                    EntityType = field_type.parameters[2]   # e.g., Vertex
                elseif field_type <: Tuple
                    # Legacy format: Tuple{QuantityType, EntityType}
                    QuantityType = field_type.parameters[1]
                    EntityType = field_type.parameters[2]
                else
                    error("Unknown field type format: $field_type (expected DOF{...} or Tuple{...})")
                end
                
                # Determine component count from QuantityType
                # QuantityType might be Displacement{3} (AbstractField) or Vec{3} or Float64
                # Use quantity_type to get underlying quantity type, then dof_size
                if QuantityType <: AbstractField
                    # Displacement{3} → Vec{3} via quantity_type trait
                    Q = quantity_type(QuantityType)  # Vec{3}
                    n_components = dof_size(Q)  # dof_size(Vec{3}) = 3
                else
                    # Already a quantity type (Vec{3}, Float64, etc.)
                    n_components = dof_size(QuantityType)
                end
                
                # Decode entity index and component
                entity_idx = div(dof_in_field - 1, n_components) + 1
                component = mod(dof_in_field - 1, n_components) + 1
                
                # For Vertex entities, node_idx = entity_idx
                node_idx = (EntityType == Vertex) ? entity_idx : entity_idx
                
                return DOFFieldInfo(field_name, EntityType, entity_idx, component, node_idx)
            end
            
            dof_offset = range_end
        end
        
        error("Local DOF index $local_dof_idx out of range for element")
    else
        # Single-field case: S is a DOF type directly
        # For now, assume it's Displacement{3} at Vertex
        # This is the most common case
        
        # Get topology to determine number of nodes
        K = topology_type(element)
        n_nodes = nnodes(K())  # nnodes accepts instance, not type
        
        # Assume 3 components (Vec{3})
        n_components = 3
        
        # Decode: local_dof_idx maps to (node_idx, component)
        node_idx = div(local_dof_idx - 1, n_components) + 1
        component = mod(local_dof_idx - 1, n_components) + 1
        
        if node_idx > n_nodes
            error("Local DOF index $local_dof_idx exceeds element size (max: $(n_nodes * n_components))")
        end
        
        return DOFFieldInfo(:u, Vertex, node_idx, component, node_idx)
    end
end

"""
    flatten_dof_indices(dofs::NTuple{N, UInt64}) -> Vector{Int}

Convert DOF tuple to vector for iteration.

# Arguments
- `dofs`: Tuple of global DOF indices

# Returns
- Vector of Int (converted from UInt64)
"""
function flatten_dof_indices(dofs::NTuple{N, UInt64}) where {N}
    return [Int(d) for d in dofs]
end

"""
    fill_dof_buffer!(buffer::Vector{Int}, dofs::NTuple{N, UInt64}) -> Int

Fill pre-allocated buffer with DOF indices from tuple.

Zero-allocation alternative to `flatten_dof_indices()`.

# Arguments
- `buffer`: Pre-allocated buffer (must have length >= N)
- `dofs`: Tuple of global DOF indices

# Returns
- Number of DOFs filled (N)

# Example
```julia
buffer = Vector{Int}(undef, 24)  # Pre-allocate for max element DOFs
n = fill_dof_buffer!(buffer, element.dof_indices)
for i in 1:n
    dof = buffer[i]
    # Process...
end
```
"""
@inline function fill_dof_buffer!(buffer::Vector{Int}, dofs::NTuple{N, UInt64}) where {N}
    @inbounds for i in 1:N
        buffer[i] = Int(dofs[i])
    end
    return N
end

# Note: Functions are available in the assemblers module scope
# They will be used by dof_based_coo.jl via include()
