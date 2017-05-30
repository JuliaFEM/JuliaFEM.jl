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

""" Convert vector of Int8 to ASCII string. """
function to_ascii(data::Vector{Int8})
    return ascii(unsafe_string(pointer(convert(Vector{UInt8}, data))))
end

function get_mesh(med::MEDFile, mesh_name::String)
    if !haskey(med.data["FAS"], mesh_name)
        warn("Mesh $mesh_name not found from med file.")
        meshes = get_mesh_names(med)
        all_meshes = join(meshes, ", ")
        warn("Available meshes: $all_meshes")
        error("Mesh $mesh_name not found.")
    end
    return med.data["FAS"][mesh_name]
end

""" Return node sets from med file.

Notes
-----
One node set id can have multiple names.

"""
function get_node_sets(med::MEDFile, mesh_name::String)::Dict{Int64, Vector{String}}
    mesh = get_mesh(med, mesh_name)
    node_sets = Dict{Int64, Vector{String}}(0 => ["NALL"])
    if !haskey(mesh, "NOEUD")
        return node_sets
    end
    for (k, v) in mesh["NOEUD"]
        nset_id = parse(Int, split(k, "_")[2])
        node_sets[nset_id] = collect(to_ascii(d) for d in v["GRO"]["NOM"])
    end
    return node_sets
end

""" Return element sets from med file.

Notes
-----
One element set id can have multiple names.

"""
function get_element_sets(med::MEDFile, mesh_name::String)::Dict{Int64, Vector{String}}
    mesh = get_mesh(med, mesh_name)
    element_sets = Dict{Int64, Vector{String}}()
    if !haskey(mesh, "ELEME")
        return element_sets
    end
    for (k, v) in mesh["ELEME"]
        elset_id = parse(Int, split(k, '_')[2])
        element_sets[elset_id] = collect(to_ascii(d) for d in v["GRO"]["NOM"])
    end
    return element_sets
end

function get_nodes(med::MEDFile, nsets::Dict{Int, Vector{String}}, mesh_name::String)
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
    d = Dict{Int64}{Tuple{Vector{String}, Vector{Float64}}}()
    for i=1:nnodes
        nset = nsets[nset_ids[i]]
        d[node_ids[i]] = (nset, node_coords[:, i])
    end
    return d
end

function get_connectivity(med::MEDFile, elsets::Dict{Int64, Vector{String}}, mesh_name::String)
    if !haskey(elsets, 0)
        elsets[0] = ["OTHER"]
    end
    increments = keys(med.data["ENS_MAA"][mesh_name])
    @assert length(increments) == 1
    increment = first(increments)
    all_elements = med.data["ENS_MAA"][mesh_name][increment]["MAI"]
    d = Dict{Int64, Tuple{Symbol, Vector{String}, Vector{Int64}}}()
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
            elset = elsets[elset_ids[i]]
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
function parse_aster_med_file(fn, mesh_name=nothing)
    med = MEDFile(fn)
    mesh_names = get_mesh_names(med::MEDFile)
    all_meshes = join(mesh_names, ", ")
    if mesh_name == nothing
        length(mesh_names) == 1 || error("several meshes found from med, pick one: $all_meshes")
        mesh_name = mesh_names[1]
    else
        mesh_name in mesh_names || error("Mesh $mesh_name not found from mesh file $fn. Available meshes: $all_meshes")
    end

    debug("Code Aster .med reader info:")
    elsets = get_element_sets(med, mesh_name)
    for (k,v) in elsets
        debug("ELSET $k => $v")
    end
    nsets = get_node_sets(med, mesh_name)
    for (k,v) in nsets
        debug("NSET $k => $v")
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
    :Pyr5   => [1,4,3,2,5],
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

    :PY5 => :Pyr5,
    :P13 => :Pyr13,

    )

""" Read code aster mesh and return Mesh. """
function aster_read_mesh(fn, mesh_name=nothing; reorder_element_connectivity=true)
    result = parse_aster_med_file(fn, mesh_name)
    mesh = Mesh()
    for (nid, (nsets, ncoords)) in result["nodes"]
        add_node!(mesh, nid, ncoords)
        for nset in nsets
            add_node_to_node_set!(mesh, Symbol(nset), nid)
        end
    end
    for (elid, (eltype, elsets, elcon)) in result["connectivity"]
        haskey(mapping, eltype) || error("Code Aster .med reader: element type $eltype not found from mapping")
        add_element!(mesh, elid, mapping[eltype], elcon)
        for elset in elsets
            add_element_to_element_set!(mesh, Symbol(elset), elid)
        end
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
    stripper(node_name) = strip(ascii(unsafe_string(pointer(convert(Vector{UInt8}, node_name)))))
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

