# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using HDF5
using JuliaFEM

function aster_parse_nodes(section; strip_characters=true)
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

function parse(mesh, ::Type{Val{:CODE_ASTER_MAIL}})
    model = Dict()
    header = nothing
    data = []
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


""" Code Aster binary file (.med). """
type MEDFile
    data :: Dict
end

function MEDFile(fn::String)
    return MEDFile(h5read(fn, "/"))
end

function get_mesh_names(med::MEDFile)
    return sort(collect(keys(med.data["FAS"])))
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
    if !haskey(med.data["FAS"], mesh_name)
        warn("Mesh $mesh_name not found from med file.")
        meshes = get_mesh_names(med)
        all_meshes = join(meshes, ", ")
        warn("Available meshes: $all_meshes")
        error("Mesh $mesh_name not found.")
    end
    haskey(med.data["FAS"][mesh_name], "NOEUD") || return ns
    nsets = med.data["FAS"][mesh_name]["NOEUD"]
    for nset in keys(nsets)
        k = split(nset, "_")
        nset_id = parse(Int, k[2])
        nset_name = ascii(unsafe_string(pointer(convert(Vector{UInt8}, nsets[nset]["GRO"]["NOM"][1]))))
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
        elset_name = ascii(unsafe_string(pointer(convert(Vector{UInt8}, elsets[elset]["GRO"]["NOM"][1]))))
        es[elset_id] = Symbol(elset_name)
    end
    return es
end

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
            d[element_ids[i]] = (eltype, elset, elco)
        end
    end
    return d
end

""" Parse code aster .med file.

Paramters
---------
fn
    file name to parse
mesh_name :: optional
    mesh name, if several meshes in one file

Returns
-------
Dict containing fields "nodes" and "connectivity".

"""
function parse_aster_med_file(fn, mesh_name=nothing; debug=false)
    med = MEDFile(fn)
    mesh_names = get_mesh_names(med::MEDFile)
    all_meshes = join(mesh_names, ", ")
    if mesh_name == nothing
        length(mesh_names) == 1 || error("several meshes found from med, pick one: $all_meshes")
        mesh_name = mesh_names[1]
    else
        mesh_name in mesh_names || error("Mesh $mesh_name not found from mesh file $fn. Available meshes: $all_meshes")
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
    result = Dict("nodes" => nodes, "connectivity" => conn)
    result["nodes"] = nodes
    result["connectivity"] = conn
    return result
end

# some glues about ordering, this is still a mystery..
# http://onelab.info/pipermail/gmsh/2008/003850.html
# http://caelinux.org/wiki/index.php/Proj:UNVConvert

#global const med_connectivity = Dict{Symbol, Vector{Int}}(
#    :Tet4 => [3, 2, 1, 4],
#    :Hex8 => [4, 8, 7, 3, 1, 5, 6, 2],  # ..?
#    :Tet10 => [3, 2, 1, 4, 6, 5, 7, 10, 9, 8])

global const med_connectivity = Dict{Symbol, Vector{Int}}(
    :Tet4   => [4,3,1,2],
    :Tet10  => [4,3,1,2,10,7,8,9,6,5],
    :Wedge6 => [4,5,6,1,2,3],
    :Hex8   => [4,8,7,3,1,5,6,2],
    :Hex20  => [4,8,7,3,1,5,6,2,20,15,19,11,12,16,14,10,17,13,18,9],
    :Hex27  => [4,8,7,3,1,5,6,2,20,15,19,11,12,16,14,10,17,13,18,9,24,25,26,23,21,22,27])

# element names in CA -> element names in JuliaFEM
global const mapping = Dict(

    :PO1 => :Poi1,

    :SE2 => :Seg2,
    :SE3 => :Seg3,
    :SE4 => :Seg4,

    :TR3 => :Tri3,
    :TR6 => :Tri6,
    :TR7 => :Tri7,

    :QU4 => :Quad4,
    :QU8 => :Quad8,
    :QU9 => :Quad9,

    :TE4 => :Tet4,
    :T10 => :Tet10,

    :PE6 => :Wedge6,
    :P15 => :Wedge15,
    :P18 => :Wedge18,

    :HE8 => :Hex8,
    :H20 => :Hex20,
    :H27 => :Hex27,

    :PY5 => :Pyramid5,
    :P13 => :Pyramid13,

    )

function aster_read_mesh(fn, mesh_name=nothing; reorder_element_connectivity=true)
    result = parse_aster_med_file(fn, mesh_name)
    mesh = Mesh()
    for (nid, (nset, ncoords)) in result["nodes"]
        add_node!(mesh, nid, ncoords)
        add_node_to_node_set!(mesh, nset, nid)
    end
    for (elid, (eltype, elset, elcon)) in result["connectivity"]
        haskey(mapping, eltype) || error("Code Aster .med reader: element type $eltype not found from mapping")
        add_element!(mesh, elid, mapping[eltype], elcon)
        add_element_to_element_set!(mesh, elset, elid)
    end
    if reorder_element_connectivity
        reorder_element_connectivity!(mesh, med_connectivity)
    end
    return mesh
end

""" Code Aster result file (.rmed). """
type RMEDFile
    data :: Dict
end

function RMEDFile(fn::String)
    return RMEDFile(h5read(fn, "/"))
end

""" Return nodes from result med file. """
function aster_read_nodes(rmed::RMEDFile)
    increments = keys(rmed.data["ENS_MAA"]["MAIL"])
    @assert length(increments) == 1
    increment = first(increments)
    nodes = rmed.data["ENS_MAA"]["MAIL"][increment]["NOE"]
    node_names = nodes["NOM"]
    node_coords = nodes["COO"]
    nnodes = length(node_names)
    dim = round(Int, length(node_coords)/nnodes)
    node_coords = reshape(node_coords, nnodes, dim)'
    stripper(node_name) = strip(ascii(pointer(convert(Vector{UInt8}, node_name))))
    node_names = map(stripper, node_names)
    # INFO: quite safe assumption is that id is in node name, i.e. N1 => 1, N123 => 123
    node_id(node_name) = parse(matchall(r"\d+", node_name)[1])
    node_ids = map(node_id, node_names)
    nodes = Dict(j => node_coords[:,j] for j in node_ids)
    return nodes
end

""" Read nodal field from rmed file. """
function aster_read_data(rmed::RMEDFile, field_name; field_type=:NODE,
                         info_fields=true, node_ids=nothing)

    if contains(field_name, "ELGA")
        field_type = :GAUSS
    end

    if node_ids == nothing
        nodes = aster_read_nodes(rmed)
        node_ids = sort(collect(keys(nodes)))
    end

    if info_fields
        field_names = keys(rmed.data["CHA"])
        all_fields = join(field_names, ", ")
        info("results: $all_fields")
    end

    chdata = rmed.data["CHA"]["RESU____$field_name"]
    @assert length(chdata) == 1
    increment = chdata[first(keys(chdata))]
    if field_type == :NODE
        data = increment["NOE"]["MED_NO_PROFILE_INTERNAL"]["CO"]
        results = Dict(j => data[j] for j in node_ids)
    else
        error("Unable to read result of type $field_type: not implemented")
    end
    return results
end

