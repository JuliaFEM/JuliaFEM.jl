# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
    Mesh{N, T<:AbstractTopology{N}}

Parametric mesh structure with single topology type for type stability and GPU optimization.

# Type Parameters
- `N`: Number of nodes per element (declared first, used in T constraint)
- `T<:AbstractTopology{N}`: Concrete topology type (Tet4, Tet10, Hex8, Tri6, Seg2, etc.)

# Fields

## Core Data
- `nodes::Vector{Vec{3,Float64}}`: Nodal coordinates (3D always, 2D uses z=0)
- `connectivity::Vector{NTuple{N,UInt32}}`: Fixed-size connectivity where N equals nnodes(T)
- `element_sets::Dict{Symbol,Set{UInt32}}`: Named element groups (element indices)
- `node_sets::Dict{Symbol,Set{UInt32}}`: Named node groups (node indices)
- `inverse_connectivity::Vector{Vector{Tuple{UInt32,UInt8}}}`: For each node: [(elem_id, local_idx), ...]

## Ordering and Bandwidth Optimization
- `node_permutation::Vector{UInt32}`: Maps original → reordered (identity until RCM applied)
- `node_inverse_permutation::Vector{UInt32}`: Maps reordered → original
- `element_permutation::Vector{UInt32}`: Maps original → reordered (for cache optimization)

## Naming and Industrial Workflows
- `node_ids::Dict{Union{Int,Symbol},UInt32}`: Named node lookup (e.g., :corner_node => 42, 10_000_001 => 1)
- `element_ids::Dict{Union{Int,Symbol},UInt32}`: Named element lookup (e.g., :E1 => 1, 20_000_001 => 1)

## Parallel Computing and Domain Decomposition
- `node_colors::Vector{UInt32}`: Color per node (0 = uncolored, for load balancing)
- `element_colors::Vector{UInt32}`: Color per element (0 = uncolored, for thread-safe assembly)
- `ghost_nodes::Set{UInt32}`: Nodes owned by other processes (MPI domain decomposition)
- `ghost_elements::Set{UInt32}`: Elements owned by other processes

# Design Rationale
- **Type Stability**: Mesh{Tet10} is fully concrete (10× faster than abstract mesh)
- **Fixed-Size Connectivity**: NTuple{10,Int} enables matrix reinterpretation for GPU
- **Real Workflows**: Matches multi-body assemblies (separate mesh per component)
- **GPU-Ready**: connectivity_matrix() provides zero-copy Matrix{Int} for GPU transfer
- **Bandwidth Optimization**: Node permutation for RCM/Cuthill-McKee minimizes matrix bandwidth
- **Industrial CAE**: Node/element IDs support multi-part assemblies (part1: 10M+ nodes, part2: 20M+ nodes)
- **Parallel-Ready**: Colors and ghost data structures prepared for threading and MPI

# Examples

```julia
# Create simple Tet4 mesh
nodes = [Vec(0.0, 0.0, 0.0), Vec(1.0, 0.0, 0.0), 
         Vec(0.0, 1.0, 0.0), Vec(0.0, 0.0, 1.0)]
connectivity = [(1, 2, 3, 4)]
mesh = Mesh{Tet4}(nodes, connectivity)

# Add element sets
mesh = Mesh{Tet4}(nodes, connectivity, 
                  element_sets=Dict(:all => Set(1)))

# Multi-body assembly
block = Mesh{Tet10}(...)
beams = Mesh{Seg2}(...)
assembly = Assembly(meshes=Dict(:block => block, :beams => beams))
```

See also: [`connectivity_matrix`](@ref), [`extract_surface`](@ref), [`Assembly`](@ref)
"""
struct Mesh{N,T<:AbstractTopology{N}} <: AbstractMesh
    # Core data
    nodes::Vector{Vec{3,Float64}}
    connectivity::Vector{NTuple{N,UInt32}}
    element_sets::Dict{Symbol,Set{UInt32}}
    node_sets::Dict{Symbol,Set{UInt32}}
    inverse_connectivity::Vector{Vector{Tuple{UInt32,UInt8}}}

    # Ordering and bandwidth optimization
    node_permutation::Vector{UInt32}
    node_inverse_permutation::Vector{UInt32}
    element_permutation::Vector{UInt32}

    # Naming and industrial workflows
    node_ids::Dict{Union{Int,Symbol},UInt32}
    element_ids::Dict{Union{Int,Symbol},UInt32}

    # Parallel computing
    node_colors::Vector{UInt32}
    element_colors::Vector{UInt32}
    ghost_nodes::Set{UInt32}
    ghost_elements::Set{UInt32}

    # Inner constructor with validation
    function Mesh{N,T}(
        nodes::Vector{Vec{3,Float64}},
        connectivity::Vector{NTuple{N,UInt32}},
        element_sets::Dict{Symbol,Set{UInt32}}=Dict{Symbol,Set{UInt32}}(),
        node_sets::Dict{Symbol,Set{UInt32}}=Dict{Symbol,Set{UInt32}}()
    ) where {N,T<:AbstractTopology{N}}
        # Validate connectivity size matches topology
        expected_nodes = nnodes(T())
        @assert N == expected_nodes "Connectivity tuple size ($N) must match nnodes($T) = $expected_nodes"

        # Validate all node indices are in range
        n_nodes = length(nodes)
        for (i, conn) in enumerate(connectivity)
            for node_id in conn
                @assert 1 ≤ node_id ≤ n_nodes "Element $i: node index $node_id out of range [1, $n_nodes]"
            end
        end

        # Validate element sets
        n_elements = length(connectivity)
        for (set_name, elem_ids) in element_sets
            for elem_id in elem_ids
                @assert 1 ≤ elem_id ≤ n_elements "Element set $set_name: element $elem_id out of range [1, $n_elements]"
            end
        end

        # Validate node sets
        for (set_name, node_ids) in node_sets
            for node_id in node_ids
                @assert 1 ≤ node_id ≤ n_nodes "Node set $set_name: node $node_id out of range [1, $n_nodes]"
            end
        end

        # Build inverse connectivity for nodal assembly
        # For each node, store list of (element_id, local_node_index) pairs
        inverse_connectivity = [Vector{Tuple{UInt32,UInt8}}() for _ in 1:n_nodes]
        for (elem_id, elem_conn) in enumerate(connectivity)
            for (local_idx, node_id) in enumerate(elem_conn)
                push!(inverse_connectivity[node_id], (UInt32(elem_id), UInt8(local_idx)))
            end
        end

        # Initialize ordering (identity permutations until RCM/reordering applied)
        node_permutation = collect(UInt32(1):UInt32(n_nodes))
        node_inverse_permutation = collect(UInt32(1):UInt32(n_nodes))
        element_permutation = collect(UInt32(1):UInt32(n_elements))

        # Initialize naming (empty until user assigns IDs)
        node_ids = Dict{Union{Int,Symbol},UInt32}()
        element_ids = Dict{Union{Int,Symbol},UInt32}()

        # Initialize parallel data (uncolored, no ghosts until partitioning)
        node_colors = zeros(UInt32, n_nodes)       # 0 = uncolored
        element_colors = zeros(UInt32, n_elements)  # 0 = uncolored
        ghost_nodes = Set{UInt32}()
        ghost_elements = Set{UInt32}()

        new{N,T}(nodes, connectivity, element_sets, node_sets, inverse_connectivity,
            node_permutation, node_inverse_permutation, element_permutation,
            node_ids, element_ids,
            node_colors, element_colors, ghost_nodes, ghost_elements)
    end
end

# Positional constructor for convenience (backward compatibility)
function Mesh{T}(
    nodes::Vector{Vec{3,Float64}},
    connectivity::Vector{NTuple{N,UInt32}};
    element_sets::Dict{Symbol,Set{UInt32}}=Dict{Symbol,Set{UInt32}}(),
    node_sets::Dict{Symbol,Set{UInt32}}=Dict{Symbol,Set{UInt32}}()
) where {N,T<:AbstractTopology{N}}
    Mesh{N,T}(nodes, connectivity, element_sets, node_sets)
end

# ============================================================================
# Basic Accessors
# ============================================================================

"""
    topology_type(mesh::Mesh{T}) -> Type{T}

Get the topology type of the mesh.
"""
topology_type(::Mesh{T}) where T = T

"""
    nnodes_per_element(mesh::Mesh{T}) -> Int

Get the number of nodes per element in the mesh.
"""
nnodes_per_element(mesh::Mesh{T}) where T = nnodes(T)

"""
    nelements(mesh::Mesh) -> Int

Get the number of elements in the mesh.
"""
nelements(mesh::Mesh) = length(mesh.connectivity)

"""
    nnodes_total(mesh::Mesh) -> Int

Get the total number of nodes in the mesh.
"""
nnodes_total(mesh::Mesh) = length(mesh.nodes)

# ============================================================================
# Inverse Connectivity (for Nodal Assembly)
# ============================================================================

"""
    get_elements_for_node(mesh::Mesh, node_id::UInt32) -> Vector{Tuple{UInt32,UInt8}}

Get all elements connected to a node with their local node indices.

Returns a vector of (element_id, local_node_index) tuples, where:
- `element_id`: Global element index
- `local_node_index`: Position of this node in the element's connectivity (1-based)

This is essential for nodal assembly, where we iterate over nodes and need to know
which elements contribute to each node.

# Example
```julia
mesh = Mesh{Tet10}(nodes, connectivity)
# Get all elements touching node 5
elems = get_elements_for_node(mesh, UInt32(5))
# elems might be: [(1, 3), (2, 1), (5, 7)]
# Meaning: node 5 is the 3rd node in element 1, 1st node in element 2, etc.

# Nodal assembly pattern:
for node_i in 1:nnodes_total(mesh)
    w_local = zero(Vec{3})
    for (elem_id, local_i) in get_elements_for_node(mesh, UInt32(node_i))
        elem_conn = mesh.connectivity[elem_id]
        for (local_j, node_j) in enumerate(elem_conn)
            K_ij = compute_stiffness_block(elem_id, local_i, local_j)
            v_j = get_dof(v, node_j)
            w_local += K_ij ⊡ v_j
        end
    end
    set_dof!(w, node_i, w_local)
end
```

See also: [`nnodes_per_element`](@ref), [`get_node`](@ref)
"""
function get_elements_for_node(mesh::Mesh, node_id::UInt32)
    @assert 1 ≤ node_id ≤ nnodes_total(mesh) "Node index $node_id out of range [1, $(nnodes_total(mesh))]"
    return mesh.inverse_connectivity[node_id]
end

"""
    get_elements_for_node(mesh::Mesh, node_id::Int) -> Vector{Tuple{UInt32,UInt8}}

Convenience wrapper accepting Int node_id (converts to UInt32).
"""
get_elements_for_node(mesh::Mesh, node_id::Int) = get_elements_for_node(mesh, UInt32(node_id))

# ============================================================================
# Connectivity Matrix (GPU-Ready)
# ============================================================================

"""
    connectivity_matrix(mesh::Mesh{T}) -> Matrix{UInt32}

Convert connectivity to a dense matrix for GPU transfer.

Returns a matrix of size (nnodes(T), nelements(mesh)) where each column
contains the node indices for one element. This enables efficient GPU
transfer and BLAS/LAPACK operations.

# Example
```julia
mesh = Mesh{Tet10}(nodes, connectivity)  # 1000 elements
conn_mat = connectivity_matrix(mesh)     # 10×1000 matrix
gpu_conn = CuArray(conn_mat)             # Single contiguous transfer!
```
"""
function connectivity_matrix(mesh::Mesh{T}) where T
    N = nnodes(T)
    n_elem = nelements(mesh)
    # Reinterpret Vector{NTuple{N,UInt32}} as flat UInt32 array, then reshape
    # This is zero-copy!
    return reshape(reinterpret(UInt32, mesh.connectivity), N, n_elem)
end

# ============================================================================
# Node Operations
# ============================================================================

"""
    get_node(mesh::Mesh, node_id::Int) -> Vec{3,Float64}

Get coordinates of a node by its index.
"""
function get_node(mesh::Mesh, node_id::Int)
    @assert 1 ≤ node_id ≤ nnodes_total(mesh) "Node index $node_id out of range"
    return mesh.nodes[node_id]
end

"""
    find_nearest_nodes(mesh::Mesh, coords::Vec{3,Float64}, npts::Int=1; node_set::Union{Nothing,Symbol}=nothing) -> Vector{UInt32}

Find the npts nearest nodes to the given coordinates.

# Arguments
- `mesh::Mesh`: The mesh
- `coords::Vec{3,Float64}`: Target coordinates
- `npts::Int=1`: Number of nearest nodes to return
- `node_set::Union{Nothing,Symbol}=nothing`: Restrict search to this node set

# Returns
- `Vector{UInt32}`: Indices of nearest nodes, sorted by distance

# Example
```julia
# Find 3 nearest nodes to point (0.5, 0.5, 0.0)
nearest = find_nearest_nodes(mesh, Vec(0.5, 0.5, 0.0), 3)

# Find nearest node in a specific node set
nearest = find_nearest_nodes(mesh, coords, 1; node_set=:boundary)
```
"""
function find_nearest_nodes(
    mesh::Mesh,
    coords::Vec{3,Float64},
    npts::Int=1;
    node_set::Union{Nothing,Symbol}=nothing
)
    @assert npts ≥ 1 "Number of points must be at least 1"

    # Build list of (node_id, distance) pairs
    distances = Tuple{UInt32,Float64}[]

    if node_set === nothing
        # Search all nodes
        for (node_id, node_coords) in enumerate(mesh.nodes)
            dist = norm(coords - node_coords)
            push!(distances, (UInt32(node_id), dist))
        end
    else
        # Search only nodes in specified set
        @assert haskey(mesh.node_sets, node_set) "Node set $node_set not found"
        for node_id in mesh.node_sets[node_set]
            node_coords = mesh.nodes[node_id]
            dist = norm(coords - node_coords)
            push!(distances, (node_id, dist))
        end
    end

    # Sort by distance and return first npts node IDs
    sort!(distances, by=x -> x[2])
    n_return = min(npts, length(distances))
    return UInt32[distances[i][1] for i in 1:n_return]
end

"""
    find_nearest_node(mesh::Mesh, coords::Vec{3,Float64}; node_set::Union{Nothing,Symbol}=nothing) -> UInt32

Find the single nearest node to the given coordinates.

Convenience wrapper around `find_nearest_nodes(mesh, coords, 1; node_set=node_set)`.
"""
function find_nearest_node(
    mesh::Mesh,
    coords::Vec{3,Float64};
    node_set::Union{Nothing,Symbol}=nothing
)
    return first(find_nearest_nodes(mesh, coords, 1; node_set=node_set))
end

# ============================================================================
# Element Set Operations
# ============================================================================

"""
    get_element_set(mesh::Mesh, set_name::Symbol) -> Set{UInt32}

Get an element set by name.
"""
function get_element_set(mesh::Mesh, set_name::Symbol)
    @assert haskey(mesh.element_sets, set_name) "Element set $set_name not found"
    return mesh.element_sets[set_name]
end

"""
    get_elements_in_set(mesh::Mesh, set_name::Symbol) -> Vector{UInt32}

Get element indices in a set as a vector (sorted).
"""
function get_elements_in_set(mesh::Mesh, set_name::Symbol)
    elem_set = get_element_set(mesh, set_name)
    return sort(collect(elem_set))
end

# ============================================================================
# Node Set Operations
# ============================================================================

"""
    get_node_set(mesh::Mesh, set_name::Symbol) -> Set{UInt32}

Get a node set by name.
"""
function get_node_set(mesh::Mesh, set_name::Symbol)
    @assert haskey(mesh.node_sets, set_name) "Node set $set_name not found"
    return mesh.node_sets[set_name]
end

"""
    get_nodes_in_set(mesh::Mesh, set_name::Symbol) -> Vector{UInt32}

Get node indices in a set as a vector (sorted).
"""
function get_nodes_in_set(mesh::Mesh, set_name::Symbol)
    node_set = get_node_set(mesh, set_name)
    return sort(collect(node_set))
end

"""
    create_node_set_from_element_set!(mesh::Mesh, elem_set_name::Symbol, node_set_name::Symbol=elem_set_name)

Create a node set containing all nodes from elements in an element set.

# Arguments
- `mesh::Mesh`: The mesh (modified in-place)
- `elem_set_name::Symbol`: Source element set name
- `node_set_name::Symbol`: Target node set name (defaults to same as element set)

# Example
```julia
# Create node set "surface" from element set "surface"
create_node_set_from_element_set!(mesh, :surface)

# Or with different names
create_node_set_from_element_set!(mesh, :volume_elements, :volume_nodes)
```
"""
function create_node_set_from_element_set!(
    mesh::Mesh,
    elem_set_name::Symbol,
    node_set_name::Symbol=elem_set_name
)
    @assert haskey(mesh.element_sets, elem_set_name) "Element set $elem_set_name not found"

    node_ids = Set{UInt32}()
    for elem_id in mesh.element_sets[elem_set_name]
        # Add all nodes from this element
        for node_id in mesh.connectivity[elem_id]
            push!(node_ids, node_id)
        end
    end

    mesh.node_sets[node_set_name] = node_ids
    @info "Created node set :$node_set_name with $(length(node_ids)) nodes from element set :$elem_set_name"
    return nothing
end

# ============================================================================
# Surface Extraction
# ============================================================================

"""
    extract_surface(mesh::Mesh{T}, face_set::Symbol) -> Mesh{FaceT}

Extract a surface mesh from volume elements.

# Arguments
- `mesh::Mesh{T}`: Volume mesh (T must be Tet4, Tet10, Hex8, Hex20, etc.)
- `face_set::Symbol`: Element set defining surface elements

# Returns
- `Mesh{FaceT}`: Surface mesh where FaceT = surface_topology(T)
  * Tet4 → Tri3
  * Tet10 → Tri6
  * Hex8 → Quad4
  * Hex20 → Quad4

# Example
```julia
volume = Mesh{Tet10}(nodes, connectivity, 
                     element_sets=Dict(:all => Set(1:100)))
surface = extract_surface(volume, :all)  # Returns Mesh{Tri6}
```
"""
function extract_surface(mesh::Mesh{T}, face_set::Symbol) where T
    FaceT = surface_topology(T)
    n_face_nodes = nnodes(FaceT)

    # Get elements in face set
    @assert haskey(mesh.element_sets, face_set) "Element set $face_set not found"
    face_elements = mesh.element_sets[face_set]

    # Extract surface connectivity (simplified - assumes first n_face_nodes form a face)
    # TODO: Proper face extraction logic based on topology
    surface_conn = NTuple{n_face_nodes,UInt32}[]
    for elem_id in face_elements
        elem_conn = mesh.connectivity[elem_id]
        face_conn = ntuple(i -> elem_conn[i], n_face_nodes)
        push!(surface_conn, face_conn)
    end

    # Reuse same nodes (surface mesh references volume nodes)
    return Mesh{FaceT}(mesh.nodes, surface_conn)
end

# ============================================================================
# Validation and Introspection
# ============================================================================

"""
    validate(mesh::Mesh) -> Bool

Validate mesh integrity (connectivity, sets, etc.).
"""
function validate(mesh::Mesh{T}) where T
    n_nodes = nnodes_total(mesh)
    n_elements = nelements(mesh)
    expected_nodes_per_elem = nnodes(T)

    # Check connectivity
    for (i, conn) in enumerate(mesh.connectivity)
        @assert length(conn) == expected_nodes_per_elem "Element $i: expected $expected_nodes_per_elem nodes, got $(length(conn))"
        for node_id in conn
            @assert 1 ≤ node_id ≤ n_nodes "Element $i: node $node_id out of range [1, $n_nodes]"
        end
    end

    # Check element sets
    for (set_name, elem_ids) in mesh.element_sets
        for elem_id in elem_ids
            @assert 1 ≤ elem_id ≤ n_elements "Element set $set_name: element $elem_id out of range"
        end
    end

    # Check node sets
    for (set_name, node_ids) in mesh.node_sets
        for node_id in node_ids
            @assert 1 ≤ node_id ≤ n_nodes "Node set $set_name: node $node_id out of range"
        end
    end

    return true
end

"""
    info(mesh::Mesh)

Print mesh information.
"""
function info(mesh::Mesh{T}) where T
    println("Mesh{$T}:")
    println("  Nodes: $(nnodes_total(mesh))")
    println("  Elements: $(nelements(mesh)) ($(nnodes(T)) nodes/element)")
    println("  Element sets: $(length(mesh.element_sets))")
    for (name, elems) in mesh.element_sets
        println("    :$name => $(length(elems)) elements")
    end
    println("  Node sets: $(length(mesh.node_sets))")
    for (name, nodes) in mesh.node_sets
        println("    :$name => $(length(nodes)) nodes")
    end
end

Base.show(io::IO, mesh::Mesh{T}) where T = print(io, "Mesh{$T}($(nnodes_total(mesh)) nodes, $(nelements(mesh)) elements)")

# ============================================================================
# Node and Element Naming (Industrial CAE Workflows)
# ============================================================================

"""
    set_node_id!(mesh::Mesh, internal_index::UInt32, id::Union{Int,Symbol})

Assign a named ID to a node. Useful for industrial workflows where nodes have
specific ID ranges (e.g., part1: 10_000_000+, part2: 20_000_000+) or symbolic
names (e.g., :corner_node, :N1).

# Examples
```julia
# Industrial ID ranges (multi-part assembly)
for i in 1:100
    set_node_id!(mesh, UInt32(i), 10_000_000 + i)  # Part 1 nodes
end

# Symbolic names (Code Aster style)
set_node_id!(mesh, UInt32(1), :N1)
set_node_id!(mesh, UInt32(42), :corner_node)
```
"""
function set_node_id!(mesh::Mesh, internal_index::UInt32, id::Union{Int,Symbol})
    @assert 1 ≤ internal_index ≤ nnodes_total(mesh) "Node index out of range"
    mesh.node_ids[id] = internal_index
    return nothing
end

"""
    get_node_by_id(mesh::Mesh, id::Union{Int,Symbol}) -> UInt32

Get internal node index from named ID.

# Example
```julia
set_node_id!(mesh, UInt32(42), :corner_node)
idx = get_node_by_id(mesh, :corner_node)  # Returns UInt32(42)
```
"""
function get_node_by_id(mesh::Mesh, id::Union{Int,Symbol})
    @assert haskey(mesh.node_ids, id) "Node ID $id not found"
    return mesh.node_ids[id]
end

"""
    set_element_id!(mesh::Mesh, internal_index::UInt32, id::Union{Int,Symbol})

Assign a named ID to an element. Similar to node IDs but for elements.

# Examples
```julia
# Industrial ID ranges
set_element_id!(mesh, UInt32(1), 20_000_001)

# Symbolic names
set_element_id!(mesh, UInt32(1), :E1)
```
"""
function set_element_id!(mesh::Mesh, internal_index::UInt32, id::Union{Int,Symbol})
    @assert 1 ≤ internal_index ≤ nelements(mesh) "Element index out of range"
    mesh.element_ids[id] = internal_index
    return nothing
end

"""
    get_element_by_id(mesh::Mesh, id::Union{Int,Symbol}) -> UInt32

Get internal element index from named ID.
"""
function get_element_by_id(mesh::Mesh, id::Union{Int,Symbol})
    @assert haskey(mesh.element_ids, id) "Element ID $id not found"
    return mesh.element_ids[id]
end

# ============================================================================
# Coloring for Parallel Assembly and Load Balancing
# ============================================================================

"""
    set_node_color!(mesh::Mesh, node_index::UInt32, color::UInt32)

Assign a color to a node. Color 0 means uncolored. Used for:
- Load balancing (assign nodes to MPI ranks)
- Identifying process ownership in domain decomposition

# Example
```julia
# Assign nodes to 4 MPI ranks
for i in 1:nnodes_total(mesh)
    rank = mod(i-1, 4) + 1  # Round-robin: 1,2,3,4,1,2,3,4,...
    set_node_color!(mesh, UInt32(i), UInt32(rank))
end
```
"""
function set_node_color!(mesh::Mesh, node_index::UInt32, color::UInt32)
    @assert 1 ≤ node_index ≤ nnodes_total(mesh) "Node index out of range"
    mesh.node_colors[node_index] = color
    return nothing
end

"""
    get_node_color(mesh::Mesh, node_index::UInt32) -> UInt32

Get the color of a node (0 = uncolored).
"""
function get_node_color(mesh::Mesh, node_index::UInt32)
    @assert 1 ≤ node_index ≤ nnodes_total(mesh) "Node index out of range"
    return mesh.node_colors[node_index]
end

"""
    set_element_color!(mesh::Mesh, elem_index::UInt32, color::UInt32)

Assign a color to an element. Color 0 means uncolored. Used for:
- Thread-safe assembly (elements with same color can be assembled in parallel)
- Graph coloring for lock-free nodal assembly

# Example
```julia
# After graph coloring algorithm
for (color, elem_ids) in colored_groups
    for elem_id in elem_ids
        set_element_color!(mesh, elem_id, color)
    end
end
```
"""
function set_element_color!(mesh::Mesh, elem_index::UInt32, color::UInt32)
    @assert 1 ≤ elem_index ≤ nelements(mesh) "Element index out of range"
    mesh.element_colors[elem_index] = color
    return nothing
end

"""
    get_element_color(mesh::Mesh, elem_index::UInt32) -> UInt32

Get the color of an element (0 = uncolored).
"""
function get_element_color(mesh::Mesh, elem_index::UInt32)
    @assert 1 ≤ elem_index ≤ nelements(mesh) "Element index out of range"
    return mesh.element_colors[elem_index]
end

"""
    get_elements_with_color(mesh::Mesh, color::UInt32) -> Vector{UInt32}

Get all elements with a specific color. Useful for parallel assembly loops.

# Example
```julia
# Parallel assembly by color
for color in 1:n_colors
    elems = get_elements_with_color(mesh, UInt32(color))
    Threads.@threads for elem_id in elems
        assemble_element!(K, mesh, elem_id)  # Thread-safe within same color
    end
end
```
"""
function get_elements_with_color(mesh::Mesh, color::UInt32)
    return [UInt32(i) for (i, c) in enumerate(mesh.element_colors) if c == color]
end

# ============================================================================
# Ghost Nodes and Elements (MPI Domain Decomposition)
# ============================================================================

"""
    mark_ghost_node!(mesh::Mesh, node_index::UInt32)

Mark a node as ghost (owned by another MPI rank). Ghost nodes are needed for
assembly at partition boundaries but are not part of the local DOF ownership.
"""
function mark_ghost_node!(mesh::Mesh, node_index::UInt32)
    @assert 1 ≤ node_index ≤ nnodes_total(mesh) "Node index out of range"
    push!(mesh.ghost_nodes, node_index)
    return nothing
end

"""
    is_ghost_node(mesh::Mesh, node_index::UInt32) -> Bool

Check if a node is a ghost node.
"""
function is_ghost_node(mesh::Mesh, node_index::UInt32)
    return node_index in mesh.ghost_nodes
end

"""
    mark_ghost_element!(mesh::Mesh, elem_index::UInt32)

Mark an element as ghost (owned by another MPI rank).
"""
function mark_ghost_element!(mesh::Mesh, elem_index::UInt32)
    @assert 1 ≤ elem_index ≤ nelements(mesh) "Element index out of range"
    push!(mesh.ghost_elements, elem_index)
    return nothing
end

"""
    is_ghost_element(mesh::Mesh, elem_index::UInt32) -> Bool

Check if an element is a ghost element.
"""
function is_ghost_element(mesh::Mesh, elem_index::UInt32)
    return elem_index in mesh.ghost_elements
end

"""
    get_local_nodes(mesh::Mesh) -> Vector{UInt32}

Get all non-ghost (locally owned) node indices.
"""
function get_local_nodes(mesh::Mesh)
    return [UInt32(i) for i in 1:nnodes_total(mesh) if !is_ghost_node(mesh, UInt32(i))]
end

"""
    get_local_elements(mesh::Mesh) -> Vector{UInt32}

Get all non-ghost (locally owned) element indices.
"""
function get_local_elements(mesh::Mesh)
    return [UInt32(i) for i in 1:nelements(mesh) if !is_ghost_element(mesh, UInt32(i))]
end

# ============================================================================
# Node Permutation (Bandwidth Minimization)
# ============================================================================

"""
    apply_node_permutation!(mesh::Mesh, permutation::Vector{UInt32})

Apply a node permutation (e.g., from RCM/Cuthill-McKee bandwidth minimization).
Updates both permutation and inverse permutation. Does NOT reorder actual node
data (nodes remain in original order, permutation is used during assembly).

# Example
```julia
# After computing RCM permutation
perm = reverse_cuthill_mckee(adjacency_matrix(mesh))
apply_node_permutation!(mesh, perm)

# Now mesh.node_permutation[i] gives reordered index for node i
# And mesh.node_inverse_permutation[j] gives original index for reordered position j
```
"""
function apply_node_permutation!(mesh::Mesh, permutation::Vector{UInt32})
    n = nnodes_total(mesh)
    @assert length(permutation) == n "Permutation size must match number of nodes"
    @assert sort(permutation) == collect(UInt32(1):UInt32(n)) "Invalid permutation"

    mesh.node_permutation .= permutation

    # Compute inverse permutation: inv_perm[perm[i]] = i
    for (i, j) in enumerate(permutation)
        mesh.node_inverse_permutation[j] = UInt32(i)
    end

    return nothing
end

"""
    apply_element_permutation!(mesh::Mesh, permutation::Vector{UInt32})

Apply an element permutation for cache-optimal memory access patterns.
"""
function apply_element_permutation!(mesh::Mesh, permutation::Vector{UInt32})
    n = nelements(mesh)
    @assert length(permutation) == n "Permutation size must match number of elements"
    @assert sort(permutation) == collect(UInt32(1):UInt32(n)) "Invalid permutation"

    mesh.element_permutation .= permutation

    return nothing
end

"""
    get_reordered_node_index(mesh::Mesh, original_index::UInt32) -> UInt32

Get the reordered (permuted) index for an original node index.
"""
function get_reordered_node_index(mesh::Mesh, original_index::UInt32)
    @assert 1 ≤ original_index ≤ nnodes_total(mesh) "Node index out of range"
    return mesh.node_permutation[original_index]
end

"""
    get_original_node_index(mesh::Mesh, reordered_index::UInt32) -> UInt32

Get the original index for a reordered (permuted) node index.
"""
function get_original_node_index(mesh::Mesh, reordered_index::UInt32)
    @assert 1 ≤ reordered_index ≤ nnodes_total(mesh) "Node index out of range"
    return mesh.node_inverse_permutation[reordered_index]
end

