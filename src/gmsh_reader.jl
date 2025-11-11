"""
Simple Gmsh mesh reader for tetrahedral meshes

Reads .msh format (ASCII version 4.1) and extracts:
- Nodes (coordinates)
- Elements (Tet4 connectivity)
- Physical groups (for boundary conditions)
"""

module GmshReader

export read_gmsh_mesh, GmshMesh

using LinearAlgebra

"""
Mesh data structure
"""
struct GmshMesh
    nodes::Matrix{Float64}              # 3 × n_nodes
    elements::Matrix{Int}               # 4 × n_elems (Tet4)
    physical_groups::Dict{String,Vector{Int}}  # Name → element/node indices
end

"""
Read Gmsh mesh file (ASCII format 4.1)
"""
function read_gmsh_mesh(filename::String)
    println("Reading Gmsh mesh: $filename")

    # Storage
    nodes = Dict{Int,Vector{Float64}}()
    elements = Vector{Tuple{Int,Int,Int,Int}}()
    physical_names = Dict{Int,String}()
    element_groups = Dict{Int,Vector{Int}}()

    open(filename, "r") do io
        while !eof(io)
            line = strip(readline(io))

            # Parse physical names
            if line == "\$PhysicalNames"
                n_names = parse(Int, readline(io))
                for _ in 1:n_names
                    parts = split(readline(io))
                    dim = parse(Int, parts[1])
                    tag = parse(Int, parts[2])
                    name = strip(parts[3], '"')
                    physical_names[tag] = name
                end
                readline(io)  # \$EndPhysicalNames
            end

            # Parse nodes (format 4.1)
            if line == "\$Nodes"
                header = split(readline(io))
                numEntityBlocks = parse(Int, header[1])
                numNodes = parse(Int, header[2])

                for _ in 1:numEntityBlocks
                    # Entity block header: entityDim entityTag parametric numNodesInBlock
                    block_header = split(readline(io))
                    numNodesInBlock = parse(Int, block_header[4])

                    # Read node IDs
                    node_ids = Int[]
                    for _ in 1:numNodesInBlock
                        push!(node_ids, parse(Int, readline(io)))
                    end

                    # Read coordinates
                    for node_id in node_ids
                        coords = split(readline(io))
                        x = parse(Float64, coords[1])
                        y = parse(Float64, coords[2])
                        z = parse(Float64, coords[3])
                        nodes[node_id] = [x, y, z]
                    end
                end
                readline(io)  # \$EndNodes
            end

            # Parse elements (format 4.1)
            if line == "\$Elements"
                header = split(readline(io))
                numEntityBlocks = parse(Int, header[1])
                numElements = parse(Int, header[2])

                for _ in 1:numEntityBlocks
                    # Entity block header: entityDim entityTag elementType numElementsInBlock
                    block_header = split(readline(io))
                    entityTag = parse(Int, block_header[2])
                    elementType = parse(Int, block_header[3])
                    numElementsInBlock = parse(Int, block_header[4])

                    for _ in 1:numElementsInBlock
                        parts = split(readline(io))
                        elem_id = parse(Int, parts[1])

                        # Tet4 element (type 4)
                        if elementType == 4
                            n1 = parse(Int, parts[2])
                            n2 = parse(Int, parts[3])
                            n3 = parse(Int, parts[4])
                            n4 = parse(Int, parts[5])
                            push!(elements, (n1, n2, n3, n4))

                            # Track physical group (entity tag is the physical group)
                            if entityTag > 0
                                if !haskey(element_groups, entityTag)
                                    element_groups[entityTag] = Int[]
                                end
                                push!(element_groups[entityTag], length(elements))
                            end
                        end
                    end
                end
                readline(io)  # \$EndElements
            end
        end
    end

    # Convert nodes to matrix (3 × n_nodes)
    n_nodes = length(nodes)
    node_matrix = zeros(3, n_nodes)
    for (node_id, coords) in nodes
        node_matrix[:, node_id] = coords
    end

    # Convert elements to matrix (4 × n_elems)
    n_elems = length(elements)
    elem_matrix = zeros(Int, 4, n_elems)
    for (i, elem) in enumerate(elements)
        elem_matrix[:, i] = [elem[1], elem[2], elem[3], elem[4]]
    end

    # Build physical groups with names
    groups = Dict{String,Vector{Int}}()
    for (tag, elem_ids) in element_groups
        name = get(physical_names, tag, "Group_$tag")
        groups[name] = elem_ids
    end

    mesh = GmshMesh(node_matrix, elem_matrix, groups)

    println("  Nodes:    $(size(node_matrix, 2))")
    println("  Elements: $(size(elem_matrix, 2))")
    println("  Physical groups: $(keys(groups))")

    return mesh
end

"""
Extract nodes on a boundary surface
"""
function get_surface_nodes(mesh::GmshMesh, surface_name::String)
    # For now, return nodes where X ≈ 0 (fixed end) or Z ≈ max (top)
    # This is a placeholder - proper implementation would track surface elements

    if surface_name == "FixedEnd"
        # Find nodes with X ≈ 0
        x_min = minimum(mesh.nodes[1, :])
        tol = 1e-6
        return findall(abs.(mesh.nodes[1, :] .- x_min) .< tol)
    elseif surface_name == "PressureSurface"
        # Find nodes with Z ≈ max
        z_max = maximum(mesh.nodes[3, :])
        tol = 1e-6
        return findall(abs.(mesh.nodes[3, :] .- z_max) .< tol)
    else
        return Int[]
    end
end

end  # module
