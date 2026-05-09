# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

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

# Design Rationale
- Type Stability: Mesh{Tet10} is fully concrete (10× faster than abstract mesh)
- Fixed-Size Connectivity: NTuple{10,Int} enables matrix reinterpretation for GPU
- Real Workflows: Matches multi-body assemblies (separate mesh per component)
- GPU-Ready: connectivity_matrix() provides zero-copy Matrix{Int} for GPU transfer

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

# Multi-body workflows: keep one `Mesh` per region (or partition), then build
# separate `Element`/handler sets per region. The old multi-mesh `Assembly`
# container lives only in `JuliaFEM.Legacy` when `JULIAFEM_ENABLE_LEGACY=1`.
```

See also: [`connectivity_matrix`](@ref), [`extract_surface`](@ref)
"""
struct Mesh{N,T<:AbstractTopology{N}} <: AbstractMesh
    nodes::Vector{Vec{3,Float64}}
    connectivity::Vector{NTuple{N,UInt32}}
    element_sets::Dict{Symbol,Set{UInt32}}
    node_sets::Dict{Symbol,Set{UInt32}}
    inverse_connectivity::Vector{Vector{Tuple{UInt32,UInt8}}}

    function Mesh{N,T}(
        nodes::Vector{Vec{3,Float64}},
        connectivity::Vector{NTuple{N,UInt32}},
        element_sets::Dict{Symbol,Set{UInt32}}=Dict{Symbol,Set{UInt32}}(),
        node_sets::Dict{Symbol,Set{UInt32}}=Dict{Symbol,Set{UInt32}}()
    ) where {N,T<:AbstractTopology{N}}
        expected_nodes = nnodes(T())
        @assert N == expected_nodes "Connectivity tuple size ($N) must match nnodes($T) = $expected_nodes"

        n_nodes = length(nodes)
        for (i, conn) in enumerate(connectivity)
            for node_id in conn
                @assert 1 ≤ node_id ≤ n_nodes "Element $i: node index $node_id out of range [1, $n_nodes]"
            end
        end

        n_elements = length(connectivity)
        for (set_name, elem_ids) in element_sets
            for elem_id in elem_ids
                @assert 1 ≤ elem_id ≤ n_elements "Element set $set_name: element $elem_id out of range [1, $n_elements]"
            end
        end

        for (set_name, node_ids_in_set) in node_sets
            for node_id in node_ids_in_set
                @assert 1 ≤ node_id ≤ n_nodes "Node set $set_name: node $node_id out of range [1, $n_nodes]"
            end
        end

        # Build inverse connectivity for nodal assembly:
        # for each node, the list of (element_id, local_node_index) pairs.
        inverse_connectivity = [Vector{Tuple{UInt32,UInt8}}() for _ in 1:n_nodes]
        for (elem_id, elem_conn) in enumerate(connectivity)
            for (local_idx, node_id) in enumerate(elem_conn)
                push!(inverse_connectivity[node_id], (UInt32(elem_id), UInt8(local_idx)))
            end
        end

        new{N,T}(nodes, connectivity, element_sets, node_sets, inverse_connectivity)
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
    topology_type(mesh::Mesh{N,T}) -> Type{T}

Get the topology type of the mesh.
"""
topology_type(::Mesh{N,T}) where {N,T<:AbstractTopology{N}} = T

"""
    nnodes_per_element(mesh::Mesh{N,T}) -> Int

Get the number of nodes per element in the mesh.
"""
nnodes_per_element(mesh::Mesh{N,T}) where {N,T<:AbstractTopology{N}} = nnodes(T)

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
    connectivity_matrix(mesh::Mesh{N,T}) -> Matrix{UInt32}

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
function connectivity_matrix(mesh::Mesh{N,T}) where {N,T<:AbstractTopology{N}}
    n_pe = nnodes(T)
    n_elem = nelements(mesh)
    # Reinterpret Vector{NTuple{N,UInt32}} as flat UInt32 array, then reshape
    # This is zero-copy!
    return reshape(reinterpret(UInt32, mesh.connectivity), n_pe, n_elem)
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

# ============================================================================
# Surface Extraction
# ============================================================================

"""
    surface_topology(::Type{<:AbstractTopology}) -> Type{<:AbstractTopology}

Topology trait giving the boundary-face topology for a volume topology.
Used by [`extract_surface`](@ref). Each supported volume type maps to its
boundary-face element type:

| Volume topology | Surface topology |
|-----------------|------------------|
| `Tetrahedron{4}`  | `Triangle{3}`      |
| `Tetrahedron{10}` | `Triangle{6}`      |
| `Hexahedron{8}`   | `Quadrilateral{4}` |
| `Hexahedron{20}`  | `Quadrilateral{8}` |
| `Hexahedron{27}`  | `Quadrilateral{9}` |

Topologies whose boundary is a mix of element types (e.g. `Wedge`,
`Pyramid`) intentionally have no method on this trait, so calling it on
them raises a `MethodError` with a clear message.
"""
function surface_topology end

surface_topology(::Type{Tetrahedron{4}})    = Triangle{3}
surface_topology(::Type{Tetrahedron{10}})   = Triangle{6}
surface_topology(::Type{Hexahedron{8}})     = Quadrilateral{4}
surface_topology(::Type{Hexahedron{20}})    = Quadrilateral{8}
surface_topology(::Type{Hexahedron{27}})    = Quadrilateral{9}

"""
    extract_surface(mesh::Mesh{N,T}, face_set::Symbol) -> Mesh{Nface,FaceT}

Extract a surface mesh from volume elements. The boundary-face topology
is looked up via [`surface_topology`](@ref), which currently supports
`Tetrahedron{4|10}` and `Hexahedron{8|20|27}` volume meshes.

# Arguments
- `mesh::Mesh{N,T}`: Volume mesh.
- `face_set::Symbol`: Element set whose elements should contribute their
  boundary face. The set must already exist in `mesh.element_sets`.

# Returns
- `Mesh{Nface,FaceT}` whose nodes alias the volume mesh's node array.

# Limitations
The current implementation takes the first `nnodes(FaceT)` connectivity
entries of each volume element as a face. This is correct only when the
caller has already arranged volume elements so that the first
`nnodes(FaceT)` nodes form the boundary face (e.g. extruded prism layers).
Topology-aware face extraction using the per-volume face tables remains a
known TODO; see the corresponding session log.
"""
function extract_surface(mesh::Mesh{N,T}, face_set::Symbol) where {N,T<:AbstractTopology{N}}
    if !hasmethod(surface_topology, Tuple{Type{T}})
        error("extract_surface: no surface_topology trait defined for $T. " *
              "Supported volume topologies: Tetrahedron{4|10}, Hexahedron{8|20|27}.")
    end
    FaceT = surface_topology(T)
    n_face_nodes = nnodes(FaceT)

    @assert haskey(mesh.element_sets, face_set) "Element set $face_set not found"
    face_elements = mesh.element_sets[face_set]

    # Extract surface connectivity. NOTE: this assumes the first
    # `n_face_nodes` entries of each volume element form a face. See the
    # function docstring; topology-aware extraction is on the backlog.
    surface_conn = NTuple{n_face_nodes,UInt32}[]
    for elem_id in face_elements
        elem_conn = mesh.connectivity[elem_id]
        face_conn = ntuple(i -> elem_conn[i], n_face_nodes)
        push!(surface_conn, face_conn)
    end

    return Mesh{FaceT}(mesh.nodes, surface_conn)
end

# ============================================================================
# Validation and Introspection
# ============================================================================

"""
    validate(mesh::Mesh) -> Bool

Validate mesh integrity (connectivity, sets, etc.).
"""
function validate(mesh::Mesh{N,T}) where {N,T<:AbstractTopology{N}}
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
    info([io::IO=stdout,] mesh::Mesh)

Print a human-readable summary of a `Mesh` (node/element counts and named
sets) to `io` (default `stdout`). The `io`-taking overload is the canonical
form; the no-`io` shorthand calls it with `stdout`.
"""
function info(io::IO, mesh::Mesh{N,T}) where {N,T<:AbstractTopology{N}}
    println(io, "Mesh{$T}:")
    println(io, "  Nodes: $(nnodes_total(mesh))")
    println(io, "  Elements: $(nelements(mesh)) ($N nodes/element)")
    println(io, "  Element sets: $(length(mesh.element_sets))")
    for (name, elems) in mesh.element_sets
        println(io, "    :$name => $(length(elems)) elements")
    end
    println(io, "  Node sets: $(length(mesh.node_sets))")
    for (name, nodes) in mesh.node_sets
        println(io, "    :$name => $(length(nodes)) nodes")
    end
end

info(mesh::Mesh) = info(stdout, mesh)

Base.show(io::IO, mesh::Mesh{N,T}) where {N,T<:AbstractTopology{N}} =
    print(io, "Mesh{$T}($(nnodes_total(mesh)) nodes, $(nelements(mesh)) elements)")
