# Nodal Assembly Data Structures
# 
# This module provides the inverse mapping needed for efficient nodal assembly:
# Given a node, find all elements touching it and the local node index within each element.

using Tensors

"""
    ElementNodeInfo

Information about how a node appears in an element.

# Fields
- `element_id::Int`: Global element ID
- `local_node_idx::Int`: Local node index within the element (1-based)
"""
struct ElementNodeInfo
    element_id::Int
    local_node_idx::Int
end

"""
    NodeToElementsMap

Inverse connectivity mapping: for each node, lists all elements touching it.

# Fields
- `node_to_elements::Vector{Vector{ElementNodeInfo}}`: For node j, gives all elements touching it
- `nnodes::Int`: Total number of nodes in mesh
- `nelements::Int`: Total number of elements in mesh

# Example
```julia
map = NodeToElementsMap(connectivity)
# Get all elements touching node 5
elements_touching_5 = map.node_to_elements[5]
for info in elements_touching_5
    println("Node 5 is local node ", info.local_node_idx, " in element ", info.element_id)
end
```
"""
struct NodeToElementsMap
    node_to_elements::Vector{Vector{ElementNodeInfo}}
    nnodes::Int
    nelements::Int
end

"""
    NodeToElementsMap(connectivity::Vector{NTuple{N,Int}}) where N

Build inverse mapping from element connectivity.

# Arguments
- `connectivity`: Vector of element connectivity tuples, e.g., [(1,2,3,4), (2,3,5,6), ...]

# Returns
- `NodeToElementsMap`: Inverse mapping structure

# Example
```julia
# Tet4 mesh with 2 elements
connectivity = [(1,2,3,4), (2,3,4,5)]
map = NodeToElementsMap(connectivity)

# Node 2 appears in both elements
@assert length(map.node_to_elements[2]) == 2
```
"""
function NodeToElementsMap(connectivity::Vector{NTuple{N,Int}}) where N
    nelements = length(connectivity)

    # Find maximum node ID to determine array size
    nnodes = maximum(maximum(conn) for conn in connectivity)

    # Pre-allocate vectors for each node
    node_to_elements = [Vector{ElementNodeInfo}() for _ in 1:nnodes]

    # Build inverse mapping
    for (elem_id, conn) in enumerate(connectivity)
        for (local_idx, global_node_id) in enumerate(conn)
            push!(node_to_elements[global_node_id],
                ElementNodeInfo(elem_id, local_idx))
        end
    end

    return NodeToElementsMap(node_to_elements, nnodes, nelements)
end

"""
    get_node_spider(map::NodeToElementsMap, node_id::Int) -> Vector{Int}

Get the "spider" of a node - all nodes that couple with it (including itself).

This is the union of all nodes in elements touching `node_id`. These are exactly
the nodes for which we need to compute 3×3 stiffness blocks.

# Arguments
- `map`: Node-to-elements mapping
- `node_id`: Node for which to find the spider

# Returns
- `spider_nodes::Vector{Int}`: Sorted unique list of node IDs in the spider

# Example
```julia
# For node j, find all nodes it couples with
spider = get_node_spider(map, j)
# Now compute K_blocks[k] for each k in spider
```
"""
function get_node_spider(map::NodeToElementsMap, node_id::Int,
    connectivity::Vector{NTuple{N,Int}}) where N
    spider = Set{Int}()

    # For each element touching this node
    for elem_info in map.node_to_elements[node_id]
        # Add all nodes in that element
        for node in connectivity[elem_info.element_id]
            push!(spider, node)
        end
    end

    return sort(collect(spider))
end

"""
    NodalStiffnessContribution{T}

Storage for nodal assembly contribution at a single node.

# Fields
- `node_id::Int`: Global node ID
- `spider_nodes::Vector{Int}`: Node IDs that couple with this node
- `K_blocks::Vector{Tensor{2,3,T}}`: 3×3 stiffness blocks for each spider node
- `f_int::Vec{3,T}`: Internal force at this node
- `f_ext::Vec{3,T}`: External force at this node

# Notes
- `K_blocks[k]` corresponds to `spider_nodes[k]`
- Diagonal block (self-coupling) is included in spider
- All quantities use Tensors.jl types (zero-allocation)
"""
struct NodalStiffnessContribution{T}
    node_id::Int
    spider_nodes::Vector{Int}
    K_blocks::Vector{Tensor{2,3,T,9}}
    f_int::Vec{3,T}
    f_ext::Vec{3,T}
end

"""
    NodalStiffnessContribution(node_id::Int, spider_nodes::Vector{Int}, ::Type{T}=Float64)

Allocate storage for nodal assembly contribution.

# Example
```julia
spider = get_node_spider(map, 5, connectivity)
contrib = NodalStiffnessContribution(5, spider, Float64)
# Now fill in K_blocks, f_int, f_ext during assembly
```
"""
function NodalStiffnessContribution(node_id::Int, spider_nodes::Vector{Int},
    ::Type{T}=Float64) where T
    nspider = length(spider_nodes)
    K_blocks = [zero(Tensor{2,3,T}) for _ in 1:nspider]
    f_int = zero(Vec{3,T})
    f_ext = zero(Vec{3,T})

    return NodalStiffnessContribution{T}(node_id, spider_nodes, K_blocks, f_int, f_ext)
end

"""
    matrix_vector_product_nodal(contrib::NodalStiffnessContribution, 
                                u::Vector{Vec{3,T}}) -> Vec{3,T}

Compute the matrix-vector product for one node using nodal assembly.

This computes: w_i = sum_j K_ij * u_j for node i

# Arguments
- `contrib`: Nodal stiffness contribution (contains K_blocks for all j in spider)
- `u`: Displacement field at all nodes (Vec{3} per node)

# Returns
- `w_i::Vec{3}`: Result of K_i * u at this node

# Example
```julia
# Assemble contribution for node i
contrib = assemble_nodal_contribution(element_set, node_i, u, time)

# Matrix-free matvec: w_i = K_i * u
w_i = matrix_vector_product_nodal(contrib, u)
```
"""
function matrix_vector_product_nodal(contrib::NodalStiffnessContribution{T},
    u::Vector{Vec{3,T}}) where T
    w = zero(Vec{3,T})

    # Loop over spider nodes (only non-zero columns)
    for (k, node_j) in enumerate(contrib.spider_nodes)
        K_ij = contrib.K_blocks[k]  # 3×3 block
        u_j = u[node_j]              # 3×1 displacement

        # Block matrix-vector product: K_ij is Tensor{2,3}, u_j is Vec{3}
        # Use regular matrix-vector multiplication (single contraction)
        w += K_ij ⋅ u_j  # Tensor{2,3} ⋅ Vec{3} → Vec{3}
    end

    return w
end

"""
    print_spider_info(map::NodeToElementsMap, node_id::Int, 
                     connectivity::Vector{NTuple{N,Int}}) where N

Print diagnostic information about a node's spider for debugging.
"""
function print_spider_info(map::NodeToElementsMap, node_id::Int,
    connectivity::Vector{NTuple{N,Int}}) where N
    println("Node $node_id Spider Analysis:")
    println("  Touches $(length(map.node_to_elements[node_id])) elements")

    for elem_info in map.node_to_elements[node_id]
        println("    Element $(elem_info.element_id): local node $(elem_info.local_node_idx)")
        println("      Connectivity: $(connectivity[elem_info.element_id])")
    end

    spider = get_node_spider(map, node_id, connectivity)
    println("  Spider has $(length(spider)) nodes: $spider")
    println("  → Need to compute $(length(spider)) 3×3 blocks")
    println("  → Diagonal block at node $node_id")
end
