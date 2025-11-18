# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
    AbstractRefineStrategy

Abstract base type for mesh refinement strategies.
"""
abstract type AbstractRefineStrategy end

"""
    LongestEdgeBisection <: AbstractRefineStrategy

Refinement strategy that splits hex elements along their longest dimension.

This is a simple octree-style refinement where each element is analyzed
to find its longest edge, and then split into two elements along that direction.

# Fields
- `levels::Int`: Number of refinement iterations (default: 1)

# Example
```julia
# Create coarse mesh with single element
nodes = [Vec(0.0, 0.0, 0.0), Vec(1.0, 0.0, 0.0), 
         Vec(1.0, 1.0, 0.0), Vec(0.0, 1.0, 0.0),
         Vec(0.0, 0.0, 1.0), Vec(1.0, 0.0, 1.0),
         Vec(1.0, 1.0, 1.0), Vec(0.0, 1.0, 1.0)]
connectivity = [(UInt32(1), UInt32(2), UInt32(3), UInt32(4),
                 UInt32(5), UInt32(6), UInt32(7), UInt32(8))]
mesh = Mesh{Hex8}(nodes, connectivity)

# Refine 2 levels
strategy = LongestEdgeBisection(2)
refined_mesh = refine(mesh, strategy)
```
"""
struct LongestEdgeBisection <: AbstractRefineStrategy
    levels::Int
end

LongestEdgeBisection() = LongestEdgeBisection(1)

"""
    refine(mesh::Mesh{Hex8}, strategy::LongestEdgeBisection) -> Mesh{Hex8}

Refine a Hex8 mesh using the longest edge bisection strategy.

# Arguments
- `mesh::Mesh{Hex8}`: Input mesh to refine
- `strategy::LongestEdgeBisection`: Refinement strategy with number of levels

# Returns
- `Mesh{Hex8}`: Refined mesh

# Algorithm
For each refinement level:
1. For each element, compute edge lengths
2. Determine longest edge direction (x, y, or z)
3. Create midpoint node and split element into two along that direction
4. Update connectivity and preserve element sets

# Example
```julia
mesh = Mesh{Hex8}(nodes, connectivity)
refined = refine(mesh, LongestEdgeBisection(2))  # 2 refinement levels
println("Original: ", nelements(mesh), " elements")
println("Refined:  ", nelements(refined), " elements")
```
"""
function refine(mesh::Mesh{Hex8}, strategy::LongestEdgeBisection)
    current_mesh = mesh

    for level in 1:strategy.levels
        current_mesh = _refine_once_hex8(current_mesh)
    end

    return current_mesh
end

"""
    _refine_once_hex8(mesh::Mesh{Hex8}) -> Mesh{Hex8}

Internal function: Perform one refinement iteration on Hex8 mesh.

For each hex element:
1. Compute element dimensions (max x, y, z extents)
2. Find longest dimension
3. Split element into two along that dimension
4. Create 4 new face nodes at the midplane (or reuse if they exist)
5. Generate two new hex elements

Hex8 node numbering (reference):
```
      8-------7
     /|      /|
    5-------6 |
    | |     | |
    | 4-----|-3
    |/      |/
    1-------2
```

Bottom face: 1-2-3-4 (z = -1)
Top face:    5-6-7-8 (z = +1)
"""
function _refine_once_hex8(mesh::Mesh{Hex8})
    old_nodes = copy(mesh.nodes)
    new_connectivity = NTuple{8,UInt32}[]

    # Dictionary to track existing nodes by position (for deduplication)
    node_map = Dict{Vec{3,Float64},UInt32}()
    for (i, node) in enumerate(old_nodes)
        node_map[node] = UInt32(i)
    end

    # Helper function to get or create a node
    function get_or_create_node(pos::Vec{3,Float64})
        # Round to avoid floating point comparison issues
        rounded_pos = Vec{3}(round.(Tuple(pos), digits=10))
        if haskey(node_map, rounded_pos)
            return node_map[rounded_pos]
        else
            push!(old_nodes, rounded_pos)
            node_id = UInt32(length(old_nodes))
            node_map[rounded_pos] = node_id
            return node_id
        end
    end

    # Track which element set each new element belongs to
    new_element_sets = Dict{Symbol,Set{UInt32}}()
    for (set_name, _) in mesh.element_sets
        new_element_sets[set_name] = Set{UInt32}()
    end

    element_counter = UInt32(0)

    # Process each element
    for (elem_idx, elem_conn) in enumerate(mesh.connectivity)
        # Get element's 8 nodes
        nodes_coords = [mesh.nodes[i] for i in elem_conn]

        # Determine longest dimension
        min_coords = minimum(nodes_coords)
        max_coords = maximum(nodes_coords)
        dims = max_coords - min_coords

        # Find longest dimension (1=x, 2=y, 3=z)
        longest_dim = argmax([dims[1], dims[2], dims[3]])

        # Split element along longest dimension
        if longest_dim == 1
            # Split along X direction
            new_elems = _split_hex8_x(elem_conn, nodes_coords, get_or_create_node)
        elseif longest_dim == 2
            # Split along Y direction
            new_elems = _split_hex8_y(elem_conn, nodes_coords, get_or_create_node)
        else
            # Split along Z direction
            new_elems = _split_hex8_z(elem_conn, nodes_coords, get_or_create_node)
        end

        # Add new elements to connectivity
        for new_elem in new_elems
            push!(new_connectivity, new_elem)
            element_counter += 1

            # Update element sets
            for (set_name, elem_set) in mesh.element_sets
                if elem_idx in elem_set
                    push!(new_element_sets[set_name], element_counter)
                end
            end
        end
    end

    # Create new mesh
    return Mesh{Hex8}(old_nodes, new_connectivity, new_element_sets, mesh.node_sets)
end

"""
    _split_hex8_x(conn, nodes, get_or_create_node) -> Tuple{NTuple{8,UInt32}, NTuple{8,UInt32}}

Split a Hex8 element along the X direction.

Creates 4 new nodes at the midplane perpendicular to X axis:
- Midpoint of edge 1-2
- Midpoint of edge 4-3
- Midpoint of edge 5-6
- Midpoint of edge 8-7

Returns two new hex elements.
"""
function _split_hex8_x(conn::NTuple{8,UInt32}, nodes::Vector{Vec{3,Float64}},
    get_or_create_node::Function)
    # Original nodes: 1-2-3-4-5-6-7-8
    n1, n2, n3, n4, n5, n6, n7, n8 = conn

    # Create 4 new midpoint nodes
    # Mid12: midpoint of edge 1-2 (bottom front)
    # Mid43: midpoint of edge 4-3 (bottom back)
    # Mid56: midpoint of edge 5-6 (top front)
    # Mid87: midpoint of edge 8-7 (top back)
    mid12 = 0.5 * (nodes[1] + nodes[2])
    mid43 = 0.5 * (nodes[4] + nodes[3])
    mid56 = 0.5 * (nodes[5] + nodes[6])
    mid87 = 0.5 * (nodes[8] + nodes[7])

    # Get or create nodes (deduplication!)
    nmid12 = get_or_create_node(mid12)
    nmid43 = get_or_create_node(mid43)
    nmid56 = get_or_create_node(mid56)
    nmid87 = get_or_create_node(mid87)

    # First element (left half)
    elem1 = (n1, nmid12, nmid43, n4, n5, nmid56, nmid87, n8)

    # Second element (right half)
    elem2 = (nmid12, n2, n3, nmid43, nmid56, n6, n7, nmid87)

    return (elem1, elem2)
end

"""
    _split_hex8_y(conn, nodes, get_or_create_node) -> Tuple{NTuple{8,UInt32}, NTuple{8,UInt32}}

Split a Hex8 element along the Y direction.

Creates 4 new nodes at the midplane perpendicular to Y axis.
"""
function _split_hex8_y(conn::NTuple{8,UInt32}, nodes::Vector{Vec{3,Float64}},
    get_or_create_node::Function)
    n1, n2, n3, n4, n5, n6, n7, n8 = conn

    # Create 4 new midpoint nodes
    # Mid21: midpoint of edge 2-1 (bottom front)
    # Mid34: midpoint of edge 3-4 (bottom back)
    # Mid65: midpoint of edge 6-5 (top front)
    # Mid78: midpoint of edge 7-8 (top back)
    mid21 = 0.5 * (nodes[2] + nodes[1])
    mid34 = 0.5 * (nodes[3] + nodes[4])
    mid65 = 0.5 * (nodes[6] + nodes[5])
    mid78 = 0.5 * (nodes[7] + nodes[8])

    # Get or create nodes (deduplication!)
    nmid21 = get_or_create_node(mid21)
    nmid34 = get_or_create_node(mid34)
    nmid65 = get_or_create_node(mid65)
    nmid78 = get_or_create_node(mid78)

    # First element (front half)
    elem1 = (n1, n2, nmid21, nmid34, n5, n6, nmid65, nmid78)

    # Second element (back half)
    elem2 = (nmid34, nmid21, n3, n4, nmid78, nmid65, n7, n8)

    return (elem1, elem2)
end

"""
    _split_hex8_z(conn, nodes, get_or_create_node) -> Tuple{NTuple{8,UInt32}, NTuple{8,UInt32}}

Split a Hex8 element along the Z direction.

Creates 4 new nodes at the midplane perpendicular to Z axis.
"""
function _split_hex8_z(conn::NTuple{8,UInt32}, nodes::Vector{Vec{3,Float64}},
    get_or_create_node::Function)
    n1, n2, n3, n4, n5, n6, n7, n8 = conn

    # Create 4 new midpoint nodes at mid-height
    # Mid15: midpoint of edge 1-5
    # Mid26: midpoint of edge 2-6
    # Mid37: midpoint of edge 3-7
    # Mid48: midpoint of edge 4-8
    mid15 = 0.5 * (nodes[1] + nodes[5])
    mid26 = 0.5 * (nodes[2] + nodes[6])
    mid37 = 0.5 * (nodes[3] + nodes[7])
    mid48 = 0.5 * (nodes[4] + nodes[8])

    # Get or create nodes (deduplication!)
    nmid15 = get_or_create_node(mid15)
    nmid26 = get_or_create_node(mid26)
    nmid37 = get_or_create_node(mid37)
    nmid48 = get_or_create_node(mid48)

    # First element (bottom half)
    elem1 = (n1, n2, n3, n4, nmid15, nmid26, nmid37, nmid48)

    # Second element (top half)
    elem2 = (nmid15, nmid26, nmid37, nmid48, n5, n6, n7, n8)

    return (elem1, elem2)
end

"""
    compute_element_volume(nodes::Vector{Vec{3,Float64}}) -> Float64

Compute approximate volume of a hex element using bounding box.

This is a quick approximation for determining element size, not exact volume.
"""
function compute_element_volume(nodes::Vector{Vec{3,Float64}})
    min_coords = minimum(nodes)
    max_coords = maximum(nodes)
    dims = max_coords - min_coords
    return dims[1] * dims[2] * dims[3]
end
