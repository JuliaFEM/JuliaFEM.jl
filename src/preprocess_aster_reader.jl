# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using HDF5
using JuliaFEM

function aster_create_elements(mesh, element_set, element_type=nothing; reverse_connectivity=false)
    elements = Element[]
    mapping = Dict(
        :SE2 => Seg2,
        :TR3 => Tri3,
        :TR6 => Tri6,
        :QU4 => Quad4,
        :HE8 => Hex8,
        :TE4 => Tet4,
        :T10 => Tet10)
    for (elid, (eltype, elset, elcon)) in mesh["connectivity"]
        elset == element_set || continue
        if !isa(element_type, Void)
            if isa(element_type, Tuple)
                if !(eltype in element_type)
                    continue
                end
            elseif eltype != element_type
                continue
            end
        end
        if reverse_connectivity
            elcon = reverse(elcon)
        end
        if !haskey(mapping, eltype)
            error("aster_create_elements: unknown element mapping $eltype")
        end
        element = Element(mapping[eltype], elcon)
        push!(elements, element)
    end
    update!(elements, "geometry", mesh["nodes"])
    return elements
end


function aster_parse_nodes(section::ASCIIString; strip_characters=true)
    nodes = Dict{Any, Vector{Float64}}()
    has_started = false
    for line in split(section, '\n')
        m = matchall(r"[\w.-]+", line)
        if (length(m) != 1) && (!has_started)
            continue
        end
        if length(m) == 1
            if (m[1] == "COOR_2D") || (m[1] == "COOR_3D")
                has_started = true
                continue
            end
            if m[1] == "FINSF"
                break
            end
        end
        if length(m) == 4
            nid = m[1]
            if strip_characters
                nid = matchall(r"\d", nid)
                nid = parse(Int, nid[1])
            end
            nodes[nid] = float(m[2:end])
        end
    end
    return nodes
end

function parse(mesh::ASCIIString, ::Type{Val{:CODE_ASTER_MAIL}})
    model = Dict{ASCIIString, Any}()
    header = nothing
    data = ASCIIString[]
    for line in split(mesh, '\n')
        length(line) != 0 || continue
        info("line: $line")
        if is_aster_mail_keyword(strip(line))
            header = parse_aster_header(line)
            empty!(data)
            continue
        end
        if line == "FINSF"
            info(data)
            header = nothing
            process_aster_section!(model, join(data, ""), header, Val{header[1]})
        end
    end
    return model
end


function aster_renumber_nodes_!(mesh, node_numbering)
    old_nodes = mesh["nodes"]
    new_nodes = typeof(old_nodes)()
    for (node_id, node_coords) in old_nodes
        new_node_id = node_numbering[node_id]
        new_nodes[new_node_id] = node_coords
    end
    mesh["nodes"] = new_nodes
    for (elid, (eltype, elset, elcon)) in mesh["connectivity"]
        new_elcon = [node_numbering[node_id] for node_id in elcon]
        mesh["connectivity"][elid] = (eltype, elset, new_elcon)
    end
end

function aster_renumber_nodes(mesh)
    nodemap = Dict{Int64,Int64}()
    for (i, nid) in enumerate(keys(mesh["nodes"]))
        nodemap[nid] = i
    end
    new_nodes = Dict{Int64, Vector{Float64}}()
    for (nid, ncoords) in mesh["nodes"]
        new_nodes[nodemap[nid]] = ncoords
    end
    function change_node_ids(old_ids::Vector{Int64})
        return Int[nodemap[nid] for nid in old_ids]
    end
    new_elements = Dict{Int64, Tuple{Symbol, Symbol, Vector{Int64}}}()
    for (elid, (eltype, elset, elcon)) in mesh["connectivity"]
        new_elements[elid] = (eltype, elset, change_node_ids(elcon))
    end
    mesh["nodes"] = new_nodes
    mesh["elements"] = new_elements
    return mesh
end

function aster_renumber_nodes!(mesh1, mesh2)

    reserved_node_ids = Set(collect(keys(mesh1["nodes"])))
    @debug info("already reserved node ids: $reserved_node_ids")
    mesh2_node_numbering = Dict{Int64, Int64}()

    # find new node ids assigned for mesh 2
    k = 1
    for node_id in sort(collect(keys(mesh2["nodes"])))
        @debug info("mesh2: processing node $node_id")
        # if node id is reserved in mesh 1, find new number
        if node_id in reserved_node_ids
            @debug info("node id conflict, $node_id already defined in mesh 1, renumbering")
            while k in reserved_node_ids
                k += 1
            end
            @debug info("mesh2: node $node_id -> $k")
            mesh2_node_numbering[node_id] = k
            push!(reserved_node_ids, k)
        else
            mesh2_node_numbering[node_id] = node_id
        end
    end
    @debug info("new node numering:")
    @debug println(mesh2_node_numbering)
    aster_renumber_nodes_!(mesh2, mesh2_node_numbering)

#=
    # create new nodes
    mesh2_old_nodes = mesh2["nodes"]
    mesh2_new_nodes = typeof(mesh2_old_nodes)()
    for (node_id, node_coords) in mesh2_old_nodes
        new_node_id = mesh2_node_numbering[node_id]
        mesh2_new_nodes[new_node_id] = node_coords
    end
    mesh2["nodes"] = mesh2_new_nodes

    # update connectivity
    for (elid, (eltype, elset, elcon)) in mesh2["connectivity"]
        new_elcon = [mesh2_node_numbering[node_id] for node_id in elcon]
        mesh2["connectivity"][elid] = (eltype, elset, new_elcon)
    end
=#

end


function aster_renumber_elements!(mesh1, mesh2)

    reserved_element_ids = Set(collect(keys(mesh1["connectivity"])))
    @debug info("already reserved element ids: $reserved_element_ids")
    mesh2_element_numbering = Dict{Int64, Int64}()

    # find new element ids assigned for mesh 2
    k = 1
    for element_id in sort(collect(keys(mesh2["connectivity"])))
        @debug info("mesh2: processing element $element_id")
        # if node id is reserved in mesh 1, find new number
        if element_id in reserved_element_ids
            @debug info("element id conflict, $element_id already defined in mesh 1, renumbering")
            while k in reserved_element_ids
                k += 1
            end
            @debug info("mesh2: element $element_id -> $k")
            mesh2_element_numbering[element_id] = k
            push!(reserved_element_ids, k)
        else
            mesh2_element_numbering[element_id] = element_id
        end
    end
    @debug info("element numbering for mesh 2:")
    @debug info(mesh2_element_numbering)

    # create new elements
    mesh2_old_elements = mesh2["connectivity"]
    mesh2_new_elements = typeof(mesh2_old_elements)()
    for (element_id, element_data) in mesh2_old_elements
        new_element_id = mesh2_element_numbering[element_id]
        mesh2_new_elements[new_element_id] = element_data
    end
    mesh2["connectivity"] = mesh2_new_elements

end


function aster_combine_meshes(mesh1, mesh2)

    # check that meshes are ready to be combined
    node_ids_mesh_1 = collect(keys(mesh1["nodes"]))
    node_ids_mesh_2 = collect(keys(mesh2["nodes"]))
    if length(intersect(node_ids_mesh_1, node_ids_mesh_2)) != 0
        error("nodes with same id number in both meshes, failed.")
    end
    element_ids_mesh_1 = collect(keys(mesh1["connectivity"]))
    element_ids_mesh_2 = collect(keys(mesh2["connectivity"]))
    if length(intersect(element_ids_mesh_1, element_ids_mesh_2)) != 0
        error("elements with same id number in both meshes, failed.")
    end
    @assert similar(mesh1) == similar(mesh2)
    @assert similar(mesh1["nodes"]) == similar(mesh2["nodes"])
    @assert similar(mesh1["connectivity"]) == similar(mesh2["connectivity"])

    new_mesh = similar(mesh1)
    new_mesh["nodes"] = similar(mesh1["nodes"])
    new_mesh["connectivity"] = similar(mesh1["connectivity"])

    for (node_id, node_coords) in mesh1["nodes"]
        new_mesh["nodes"][node_id] = node_coords
    end
    for (node_id, node_coords) in mesh2["nodes"]
        new_mesh["nodes"][node_id] = node_coords
    end
    for (element_id, element_data) in mesh1["connectivity"]
        new_mesh["connectivity"][element_id] = element_data
    end
    for (element_id, element_data) in mesh2["connectivity"]
        new_mesh["connectivity"][element_id] = element_data
    end
    return new_mesh
end


"""
Code Aster binary file (.med), which is exported from SALOME.
"""
type MEDFile
    data :: Dict
end

function MEDFile(fn::ASCIIString)
    MEDFile(h5read(fn, "/"))
end

function get_mesh_names(med::MEDFile)
    return collect(keys(med.data["FAS"]))
end

function get_nodes(med::MEDFile, nsets, mesh_name)
    increments = keys(med.data["ENS_MAA"][mesh_name])
    @assert length(increments) == 1
    increment = first(increments)
    nodes = med.data["ENS_MAA"][mesh_name][increment]["NOE"]
    node_ids = nodes["NUM"]
    nset_ids = nodes["FAM"]
    nnodes = length(node_ids)
    node_coords = nodes["COO"]
    dim = round(Int, length(node_coords)/nnodes)
    node_coords = reshape(node_coords, nnodes, dim)'
    d = Dict{Int64}{Tuple{Symbol, Vector{Float64}}}()
    for i=1:nnodes
        nset = Symbol(nsets[nset_ids[i]])
        d[node_ids[i]] = (nset, node_coords[:, i])
    end
    return d
end

function get_node_sets(med::MEDFile, mesh_name)
    ns = Dict{Int64, Symbol}(0 => :NALL)
    haskey(med.data["FAS"][mesh_name], "NOEUD") || return ns
    nsets = med.data["FAS"][mesh_name]["NOEUD"]
    for nset in keys(nsets)
        k = split(nset, "_")
        nset_id = parse(Int, k[2])
        nset_name = ascii(pointer(convert(Vector{UInt8}, nsets[nset]["GRO"]["NOM"][1])))
        ns[nset_id] = Symbol(nset_name)
    end
    return ns
end

function get_element_sets(med::MEDFile, mesh_name)
    es = Dict{Int64, Symbol}()
    if !haskey(med.data["FAS"][mesh_name], "ELEME")
        return es
    end
    elsets = med.data["FAS"][mesh_name]["ELEME"]
    for elset in keys(elsets)
        k = split(elset, '_')
        elset_id = parse(Int, k[2])
        elset_name = ascii(pointer(convert(Vector{UInt8}, elsets[elset]["GRO"]["NOM"][1])))
        es[elset_id] = Symbol(elset_name)
    end
    return es
end

global const med_elmap = Dict{Symbol, Vector{Int}}(
    :PO1 => [1],
    :SE2 => [1, 2],
    :SE3 => [1, 2, 3],
    :TR3 => [1, 2, 3],
    :QU4 => [1, 2, 3, 4],
    :TE4 => [3, 2, 1, 4],
    :TR6 => [1, 2, 3, 4, 5, 6],
    :QU8 => [1, 2, 3, 4, 5, 6, 7, 8],
    :HE8 => [4, 8, 7, 3, 1, 5, 6, 2],  # ..?
    :T10 => [3, 2, 1, 4, 6, 5, 7, 10, 9, 8])

function get_connectivity(med::MEDFile, elsets, mesh_name)
    elsets[0] = :OTHER
    increments = keys(med.data["ENS_MAA"][mesh_name])
    @assert length(increments) == 1
    increment = first(increments)
    all_elements = med.data["ENS_MAA"][mesh_name][increment]["MAI"]
    d = Dict{Int64, Tuple{Symbol, Symbol, Vector{Int64}}}()
    for eltype in keys(all_elements)
        elements = all_elements[eltype]
        elset_ids = elements["FAM"]
        element_ids = elements["NUM"]
        nelements = length(element_ids)
        element_connectivity = elements["NOD"]
        element_dim = round(Int, length(element_connectivity)/nelements)
        element_connectivity = reshape(element_connectivity, nelements, element_dim)'
        for i=1:nelements
            eltype = Symbol(eltype)
            elco = element_connectivity[:, i]
            elset = Symbol(elsets[elset_ids[i]])
            if haskey(med_elmap, eltype)
                elco = elco[med_elmap[eltype]]
            else
                warn("no element mapping info found for element type $eltype")
                warn("consider this as a warning: element may have french nodal ordering")
            end
            d[element_ids[i]] = (eltype, elset, elco)
        end
    end
    return d
end

""" Parse code aster .med file.

Paramters
---------
fn :: ASCIIString
    file name to parse
mesh_name :: ASCIIString, optional
    mesh name, if several meshes in one file

Returns
-------
Dict containing fields "nodes" and "connectivity".

"""
function parse_aster_med_file(fn::ASCIIString, mesh_name=nothing; debug=false)
    med = MEDFile(fn)
    if isa(mesh_name, Void)
        mesh_names = get_mesh_names(med::MEDFile)
        all_meshes = join(mesh_names, ", ")
        length(mesh_names) == 1 || error("several meshes found from med, pick one: $all_meshes")
        mesh_name = mesh_names[1]
    end
    nsets = get_node_sets(med, mesh_name)
    elsets = get_element_sets(med, mesh_name)
    if debug
        elset_names = join(values(elsets), ", ")
        info("Code Aster .med reader: found $(length(elsets)) element sets: $elset_names")
        nset_names = join(values(nsets), ", ")
        info("Code ASter .med reader: found $(length(nsets)) node sets: $nset_names")
    end
    nodes = get_nodes(med, nsets, mesh_name)
    conn = get_connectivity(med, elsets, mesh_name)
    result = Dict{ASCIIString, Any}()
    result["nodes"] = nodes
    result["connectivity"] = conn
    return result
end

global const mapping = Dict(
    :PO1 => :Poi1,
    :SE2 => :Seg2,
    :SE3 => :Seg3,
    :TR3 => :Tri3,
    :TR6 => :Tri6,
    :QU4 => :Quad4,
    :QU8 => :Quad8,
    :QU9 => :Quad9,
    :HE8 => :Hex8,
    :H20 => :Hex20,
    :TE4 => :Tet4,
    :T10 => :Tet10)

function aster_read_mesh(fn::ASCIIString, mesh_name=nothing)
    result = parse_aster_med_file(fn, mesh_name)
    mesh = Mesh()
    for (nid, (nset, ncoords)) in result["nodes"]
        add_node!(mesh, nid, ncoords)
        add_node_to_node_set!(mesh, string(nset), nid)
    end
    for (elid, (eltype, elset, elcon)) in result["connectivity"]
        haskey(mapping, eltype) || error("Code Aster .med reader: element type $eltype not found from mapping")
        add_element!(mesh, elid, mapping[eltype], elcon)
        add_element_to_element_set!(mesh, string(elset), elid)
    end
    return mesh
end

# TODO: refactor and remove obsolete stuff.
