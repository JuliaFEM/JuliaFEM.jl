# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using HDF5

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

function get_nodes(med::MEDFile, mesh_name)
    increments = keys(med.data["ENS_MAA"][mesh_name])
    @assert length(increments) == 1
    increment = first(increments)
    nodes = med.data["ENS_MAA"][mesh_name][increment]["NOE"]
    node_ids = nodes["NUM"]
    nnodes = length(node_ids)
    node_coords = nodes["COO"]
    dim = round(Int, length(node_coords)/nnodes)
    node_coords = reshape(node_coords, nnodes, dim)'
    d = Dict{Int64}{Vector{Float64}}()
    for i=1:nnodes
        d[node_ids[i]] = node_coords[:, i]
    end
    return d
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

# hex8 nodes rotating cw first in yz plane then x+1

global const med_elmap = Dict{Symbol, Vector{Int}}(
    :HE8 => [4, 8, 7, 3, 1, 5, 6, 2],
    :QU4 => [4, 3, 2, 1],
    :SE2 => [2, 1]
)

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
function parse_aster_med_file(fn::ASCIIString, mesh_name=nothing)
    med = MEDFile(fn)
    if isa(mesh_name, Void)
        mesh_names = get_mesh_names(med::MEDFile)
        all_meshes = join(mesh_names, ", ")
        length(mesh_names) == 1 || error("several meshes found from med, pick one: $all_meshes")
        mesh_name = mesh_names[1]
    end
    elsets = get_element_sets(med, mesh_name)
    elset_names = join(values(elsets), ", ")
    info("Found $(length(elsets)) element sets: $elset_names")
    nodes = get_nodes(med, mesh_name)
    conn = get_connectivity(med, elsets, mesh_name)
    result = Dict{ASCIIString, Any}()
    result["nodes"] = nodes
    result["connectivity"] = conn
    return result
end

