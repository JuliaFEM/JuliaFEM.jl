# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE

"""
DOFManager - Global DOF numbering and element creation.

Manages DOF assignment across mesh entities (vertices, edges, faces, cells).
Supports heterogeneous multi-field specifications.

See `src/dofs/README.md` for architecture and `docs/src/developer/` for examples.
"""

# Import DOF connectivity types (forward declaration - will be available after dof_connectivity.jl is loaded)
# Note: These are defined in dof_connectivity.jl, but we reference them here
# The actual import happens at module level in JuliaFEM.jl

# ============================================================================
# DOFManager State
# ============================================================================

"""
    DOFManager

Manages global DOF numbering across mesh entities.

# Fields
- `node_to_dofs`: Maps entity_id → DOF indices
- `total_dofs`: Total DOFs in system
- `field_specs`: Registered fields (field_name → (QuantityType, EntityType))
- `mesh`: Optional mesh reference for entity counting
- `dof_connectivity`: Inverse mapping from DOF → elements (built during create_elements!)
"""
mutable struct DOFManager
    node_to_dofs::Dict{Int, Vector{Int}}
    total_dofs::Int
    next_dof::Int  # Internal: next DOF index to assign
    field_specs::Dict{Symbol, Tuple{DataType, DataType}}  # field_name → (QuantityType, EntityType)
    mesh::Union{AbstractMesh, Nothing}  # Store mesh reference for counting entities
    dof_connectivity::Union{Any, Nothing}  # Inverse mapping: DOF → elements (DOFConnectivity type)
    
    function DOFManager()
        new(Dict{Int, Vector{Int}}(), 0, 1, Dict{Symbol, Tuple{DataType, DataType}}(), nothing, nothing)
    end
    
    function DOFManager(mesh::AbstractMesh)
        mgr = new(Dict{Int, Vector{Int}}(), 0, 1, Dict{Symbol, Tuple{DataType, DataType}}(), mesh, nothing)
        return mgr
    end
end

"""
    register_fields!(mgr::DOFManager, field_spec::Type{<:DOFSet})

Register fields from specification. Allows heterogeneous elements with overlapping fields.
"""
function register_fields!(mgr::DOFManager, field_spec::Type{<:DOFSet})
    for field_name in fieldnames(field_spec)
        FieldType = fieldtype(field_spec, field_name)
        
        # Extract (Quantity, Entity) tuple
        if FieldType <: Tuple
            QuantityType = FieldType.parameters[1]
            EntityType = FieldType.parameters[2]
            field_info = (QuantityType, EntityType)
        else
            error("Unknown field format for :$field_name")
        end
        
        # Check if field already registered
        if haskey(mgr.field_specs, field_name)
            existing = mgr.field_specs[field_name]
            if existing != field_info
                error("Field :$field_name already registered with different type: $existing vs $field_info")
            end
            # Already registered with same type - OK, skip
        else
            # Register new field
            mgr.field_specs[field_name] = field_info
        end
    end
    
    return mgr
end

"""
    allocate_dofs!(mgr::DOFManager, entity_id::Int, n_dofs::Int) → Vector{Int}

Allocate DOFs for entity. Appends if entity already has DOFs.
"""
function allocate_dofs!(mgr::DOFManager, node_id::Int, n_dofs::Int)
    if haskey(mgr.node_to_dofs, node_id)
        # Node already has DOFs - append new ones
        existing_dofs = mgr.node_to_dofs[node_id]
        new_dofs = mgr.next_dof:(mgr.next_dof + n_dofs - 1)
        append!(existing_dofs, new_dofs)
        mgr.next_dof += n_dofs
        mgr.total_dofs += n_dofs
        return existing_dofs
    else
        # First time seeing this node
        dofs = collect(mgr.next_dof:(mgr.next_dof + n_dofs - 1))
        mgr.node_to_dofs[node_id] = dofs
        mgr.next_dof += n_dofs
        mgr.total_dofs += n_dofs
        return dofs
    end
end

function get_node_dofs(mgr::DOFManager, node_id::Int)
    return get(mgr.node_to_dofs, node_id, Int[])
end

"""
    count_field_dofs(mgr::DOFManager, field_name::Symbol) → Int

Count total DOFs for specific field using registered field info and mesh entity counts.
"""
function count_field_dofs(mgr::DOFManager, field_name::Symbol)
    if !haskey(mgr.field_specs, field_name)
        error("Field :$field_name not registered in DOFManager. Available fields: $(keys(mgr.field_specs))")
    end
    
    # Get the field info (QuantityType, EntityType)
    (QuantityType, EntityType) = mgr.field_specs[field_name]
    
    # Calculate DOFs per entity based on quantity type
    dofs_per_entity = dof_size(QuantityType)
    
    # Count entities based on entity type
    n_entities = count_entities(mgr.mesh, EntityType)
    
    return dofs_per_entity * n_entities
end

function count_entities(mesh::Mesh, ::Type{Vertex})
    return length(mesh.nodes)
end

function count_entities(mesh::Mesh, ::Type{Cell})
    return length(mesh.connectivity)
end

function count_entities(mesh::Mesh{D,Topo}, ::Type{Edge}) where {D,Topo}
    edge_set = Set{NTuple{2,Int}}()
    
    for conn in mesh.connectivity
        # Get edges for this topology (returns Edge objects)
        edge_list = edges(Topo())
        
        for edge in edge_list
            # Edge.vertices contains local node indices (e.g., (1,2))
            # Map to global node indices from connectivity
            local_indices = edge.vertices
            global_indices = (conn[local_indices[1]], conn[local_indices[2]])
            # Sort to ensure consistent edge identification
            edge_nodes = tuple(sort(collect(global_indices))...)
            push!(edge_set, edge_nodes)
        end
    end
    
    return length(edge_set)
end

# ============================================================================
# DOF Assignment
# ============================================================================

"""
    _assign_element_dofs!(mgr, ::Type{DOF{T,E}}, ::Type{Topo}, connectivity)

Assign DOF indices for element given DOF type and connectivity.
Returns tuple of all element DOF indices.
"""
function _assign_element_dofs!(
    mgr::DOFManager,
    ::Type{DOF{T,E}},
    ::Type{Topo},
    connectivity::NTuple{N,<:Integer}
) where {T, E<:TopologicalEntity, Topo, N}
    # Determine DOFs per entity (node) from quantity type
    dofs_per_entity = dof_size(T)
    
    # Collect DOF indices for all entities
    all_dofs = Int[]
    
    for entity_id in connectivity
        entity_id_int = Int(entity_id)  # Convert UInt32 to Int
        if haskey(mgr.node_to_dofs, entity_id_int)
            # Entity already has DOFs
            existing = mgr.node_to_dofs[entity_id_int]
            if length(existing) >= dofs_per_entity
                # Reuse existing DOFs (assume same type for now)
                # TODO: More sophisticated matching for mixed DOF types
                append!(all_dofs, existing[1:dofs_per_entity])
            else
                # Need more DOFs - allocate additional
                entity_dofs = allocate_dofs!(mgr, entity_id_int, dofs_per_entity - length(existing))
                append!(all_dofs, mgr.node_to_dofs[entity_id_int])
            end
        else
            # First time seeing this entity - allocate DOFs
            entity_dofs = allocate_dofs!(mgr, entity_id_int, dofs_per_entity)
            append!(all_dofs, entity_dofs)
        end
    end
    
    return tuple(all_dofs...)
end

# Specialization for Cell entity: ONE DOF per element, not per vertex
function _assign_element_dofs!(
    mgr::DOFManager,
    ::Type{DOF{T,Cell}},
    ::Type{Topo},
    connectivity::NTuple{N,<:Integer},
    elem_id::Int
) where {T, Topo, N}
    # For Cell entity: allocate DOFs based on element ID, not connectivity
    dofs_per_cell = dof_size(T)
    
    # Use negative elem_id as entity_id to avoid collision with node IDs
    # (nodes are positive, cells are negative)
    cell_entity_id = -elem_id
    
    if haskey(mgr.node_to_dofs, cell_entity_id)
        # Cell already has DOFs (shouldn't happen normally)
        cell_dofs = mgr.node_to_dofs[cell_entity_id]
    else
        # Allocate DOFs for this cell
        cell_dofs = allocate_dofs!(mgr, cell_entity_id, dofs_per_cell)
    end
    
    return tuple(cell_dofs...)
end

# Specialization for Edge entity: DOFs on edges, not vertices
function _assign_element_dofs!(
    mgr::DOFManager,
    ::Type{DOF{T,Edge}},
    ::Type{Topo},
    connectivity::NTuple{N,<:Integer},
    elem_id::Int
) where {T, Topo, N}
    # For Edge entity: need to enumerate edges from connectivity
    # Use topology information to get edge connectivity
    dofs_per_edge = dof_size(T)
    
    # Get edges for this topology
    # edges(Topo()) returns SVector of Edge objects with .vertices field
    edge_list = edges(Topo())  # Returns SVector{N, Edge}
    
    all_dofs = Int[]
    
    for edge in edge_list
        # Extract vertex indices from Edge.vertices tuple
        (i, j) = edge.vertices
        
        # Get actual node IDs from connectivity
        node_i = Int(connectivity[i])
        node_j = Int(connectivity[j])
        
        # Create canonical edge ID: (min, max) to ensure uniqueness
        edge_id = (min(node_i, node_j), max(node_i, node_j))
        
        # Use hash of edge_id as entity ID (offset to avoid collision)
        # Convert hash to Int for allocate_dofs!
        entity_id = 1_000_000 + Int(hash(edge_id) % 1_000_000)
        
        if haskey(mgr.node_to_dofs, entity_id)
            # Edge already has DOFs (shared between elements)
            edge_dofs = mgr.node_to_dofs[entity_id]
        else
            # Allocate DOFs for this edge
            edge_dofs = allocate_dofs!(mgr, entity_id, dofs_per_edge)
        end
        
        append!(all_dofs, edge_dofs)
    end
    
    return tuple(all_dofs...)
end

"""
    _assign_element_dofs!(mgr, ::Type{NT}, K, conn[, elem_id]) where {NT<:DOFSet}

Multi-field DOF assignment. Returns NamedTuple of DOF indices per field.
"""
function _assign_element_dofs!(
    mgr::DOFManager,
    ::Type{NT},
    ::Type{Topo},
    connectivity::NTuple{N,<:Integer},
    elem_id::Int=0
) where {NT<:NamedTuple, Topo, N}
    # Get field names and field types
    field_names = fieldnames(NT)
    
    # Allocate DOFs for each field
    field_dofs = []
    for name in field_names
        FieldType = fieldtype(NT, name)  # Tuple{Displacement{3}, Vertex}
        
        # FieldType must be DOF{FieldType, EntityType}
        if FieldType <: DOF
            field_spec_type = FieldType.parameters[1]  # Displacement{3}
            entity_type = FieldType.parameters[2]  # Vertex
        else
            error("Expected DOF{FieldType, EntityType} for field :$name, got $FieldType. Use format: @DOFSet{field::DOF{FieldType, EntityType}}")
        end
        
        # Extract quantity type via trait (works with both DOF and Tuple)
        Q = quantity_type(FieldType)  # Vec{3} or Float64
        
        # Create equivalent DOF type for assignment
        DOFType = DOF{Q, entity_type}
        
        # Check if this entity type needs elem_id (Cell, Edge)
        if entity_type <: Cell || entity_type <: Edge
            if elem_id == 0
                error("elem_id required for Cell or Edge entity types in field :$name")
            end
            dofs = _assign_element_dofs!(mgr, DOFType, Topo, connectivity, elem_id)
        else
            dofs = _assign_element_dofs!(mgr, DOFType, Topo, connectivity)
        end
        
        push!(field_dofs, dofs)
    end
    
    # Construct NamedTuple with field names and DOF indices
    return NamedTuple{field_names}(tuple(field_dofs...))
end

# ============================================================================
# Element Creation API
# ============================================================================

"""
    create_elements!(mgr::DOFManager, ::Type{Element{K,P,S}}) → Vector{Element}

Create elements using existing DOFManager. Supports heterogeneous multi-field elements.
"""
function create_elements!(
    mgr::DOFManager,
    ::Type{Element{K,P,S}}
) where {K, P, S<:DOFSet}
    # Register fields from this element type
    register_fields!(mgr, S)
    
    # Verify mesh topology matches element topology
    if mgr.mesh !== nothing
        MeshTopo = typeof(mgr.mesh).parameters[2]
        if K != MeshTopo
            @warn "Element topology $K does not match mesh topology $MeshTopo - proceeding anyway (heterogeneous mesh?)"
        end
    end
    
    elements = Element{K,P,S}[]
    
    # Process each element
    for elem_id in 1:length(mgr.mesh.connectivity)
        conn = mgr.mesh.connectivity[elem_id]
        
        # Assign DOFs to element (multi-field version)
        dof_indices_named = _assign_element_dofs!(mgr, S, K, conn, elem_id)
        
        # Convert to UInt in each field
        field_names = fieldnames(S)
        dof_indices_uint = NamedTuple{field_names}(
            tuple([NTuple{length(dof_indices_named[name]),UInt}(UInt.(dof_indices_named[name])) 
                   for name in field_names]...)
        )
        
        # Create element with assigned DOFs
        elem = Element{K,P,S}(UInt(elem_id), dof_indices_uint)
        push!(elements, elem)
    end
    
    return elements
end

"""
    create_elements!(mesh::Mesh, ::Type{Element{K,P,S}}) → (Vector{Element}, DOFManager)

Create elements with assigned DOF indices for single-field specification.
Returns (elements, dof_manager).
"""
function create_elements!(
    mesh::Mesh{N,MeshTopo},
    ::Type{Element{K,P,S}}
) where {N, MeshTopo, K, P, S<:DOF}
    # For single-field DOF, Element type must be NamedTuple (DOFSet)
    # We need to wrap S in NamedTuple, but can't do it with type variables directly
    # Use a workaround: call the multi-field version with wrapped type
    S_wrapped = @eval @NamedTuple{dof::$S}
    return create_elements!(mesh, Element{K,P,S_wrapped})
end

# Multi-field version
function create_elements!(
    mesh::Mesh{N,MeshTopo},
    ::Type{Element{K,P,S}}
) where {N, MeshTopo, K, P, S<:NamedTuple}
    mgr = DOFManager()
    elements = Element{K,P,S}[]
    
    # Check topology match
    if K != MeshTopo
        error("Element topology $K does not match mesh topology $MeshTopo")
    end
    
    # Process each element
    for elem_id in 1:length(mesh.connectivity)
        conn = mesh.connectivity[elem_id]
        
        # Assign DOFs to element (multi-field version)
        dof_indices_named = _assign_element_dofs!(mgr, S, K, conn, elem_id)
        
        # Flatten NamedTuple into a single flat tuple for Element struct
        # Element always uses NTuple{N,UInt64} regardless of S being NamedTuple
        field_names = fieldnames(S)
        all_dofs = UInt64[]
        for name in field_names
            append!(all_dofs, UInt64.(dof_indices_named[name]))
        end
        
        # Create element with flattened DOF indices
        dof_indices_flat = NTuple{length(all_dofs),UInt64}(tuple(all_dofs...))
        elem = Element{K,P,S}(UInt(elem_id), dof_indices_flat)
        push!(elements, elem)
    end
    
    # Build inverse mapping: DOF → elements (computed directly from assigned DOFs)
    _build_dof_connectivity(elements, mgr)
    
    return elements, mgr
end


"""
    get_element_ids(mesh, element_set_name::String) → Vector{Int}

Get element IDs from named element set.
"""
function get_element_ids(mesh, element_set_name::String)
    if haskey(mesh.element_sets, element_set_name)
        return mesh.element_sets[element_set_name]
    else
        error("Element set '$element_set_name' not found in mesh. Available sets: $(keys(mesh.element_sets))")
    end
end

# Symbol version
function get_element_ids(mesh, element_set_name::Symbol)
    return get_element_ids(mesh, String(element_set_name))
end

# ============================================================================
# Inverse Mapping: DOF → Elements
# ============================================================================

"""
    _build_dof_connectivity(elements::Vector{<:AbstractElement}, mgr::DOFManager)

Build inverse mapping from DOF indices to elements.

This is called automatically during `create_elements!` after all DOFs are assigned.
The mapping is stored in `mgr.dof_connectivity` for use by DOF-based assemblers.

# Algorithm
1. Initialize: `dof_to_elements = [DOFElementConnection[] for _ in 1:n_total_dofs]`
2. Loop over elements:
   - Get element's global DOF indices: `elem.dof_indices`
   - For each (local_idx, global_dof):
     - Push `DOFElementConnection(elem_id, local_idx)` to `dof_to_elements[global_dof]`
3. Store `DOFConnectivity(dof_to_elements, n_total_dofs)` in `mgr.dof_connectivity`

# Arguments
- `elements`: Vector of elements with assigned DOF indices
- `mgr`: DOF manager (for total DOF count, stores result in mgr.dof_connectivity)

# Side Effects
- Mutates `mgr.dof_connectivity` with the inverse mapping
"""
function _build_dof_connectivity(
    elements::Vector{<:AbstractElement},
    mgr::DOFManager
)
    # Call build_dof_connectivity from dof_connectivity.jl
    # This function is available at runtime since dof_connectivity.jl is loaded after dof_manager.jl
    # but before this function is called (at runtime during create_elements!)
    # We can call it directly - it's in the same module
    mgr.dof_connectivity = build_dof_connectivity(elements, mgr)
    return nothing
end
