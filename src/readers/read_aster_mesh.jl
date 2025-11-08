# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/AsterReader.jl/blob/master/LICENSE

function aster_parse_nodes(section; strip_characters=true)
    nodes = Dict{Any, Vector{Float64}}()
    has_started = false
    for line in split(section, '\n')
        m = collect((string(m.match) for m in eachmatch(r"[\w.-]+", line)))
        if length(m) == 1
            if (m[1] == "COOR_2D") || (m[1] == "COOR_3D")
                has_started = true
                continue
            end
            if m[1] == "FINSF"
                break
            end
        end
        if !has_started # we have not found COOR_2D or COOR_3D yet.
            continue
        end
        to(T, x) = map(x -> parse(T, x), x)
        if length(m) == 4
            if strip_characters
                nodes[parse_node_id(m[1])] = to(Float64, m[2:end])
            else
                nodes[nid] = to(Float64, m[2:end])
            end
        end
    end
    return nodes
end


""" Code Aster binary file (.med). """
mutable struct MEDFile
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
        @warn("Mesh $mesh_name not found from med file.")
        meshes = get_mesh_names(med)
        all_meshes = join(meshes, ", ")
        @warn("Available meshes: $all_meshes")
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
    node_sets = Dict{Int64, Vector{String}}(0 => ["OTHER"])
    if !haskey(mesh, "NOEUD")
        return node_sets
    end
    for (k, v) in mesh["NOEUD"]
        nset_id = parse(Int, split(k, "_")[2])
        node_sets[nset_id] = collect(to_ascii(d) for d in v["GRO"]["NOM"])
    end
    return node_sets
end

"""
    get_element_sets(med, mesh_name)

Return element sets from med file. Return type is a dictionary, where the key is
the element set id number (integer) and value is a vector of strings, containing
human-readable name for element set.

# Notes

One element set id can have multiple names.

"""
function get_element_sets(med::MEDFile, mesh_name::String)::Dict{Int64, Vector{String}}
    mesh = get_mesh(med, mesh_name)
    element_sets = Dict{Int64, Vector{String}}()
    if !haskey(mesh, "ELEME")
        return element_sets
    end
    elset_keys = sort(collect(keys(mesh["ELEME"])))
    for (i, k) in enumerate(elset_keys)
        v = mesh["ELEME"][k]
        if startswith(k, "FAM") # exported from Salome
            elset_id = parse(Int, split(k, '_')[2])
        else # exported from Gmsh
            elset_id = -i
        end
        if !isempty(v)
            element_sets[elset_id] = map(strip, collect(to_ascii(d) for d in v["GRO"]["NOM"]))
        else
            element_sets[elset_id] = [""]
        end
    end
    return element_sets
end

function get_nodes(med::MEDFile, nsets::Dict{Int, Vector{String}}, mesh_name::String)
    increments = keys(med.data["ENS_MAA"][mesh_name])
    @assert length(increments) == 1
    increment = first(increments)
    nodes = med.data["ENS_MAA"][mesh_name][increment]["NOE"]
    nset_ids = nodes["FAM"]
    nnodes = length(nset_ids)
    node_ids = get(nodes, "NUM", collect(1:nnodes))
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
        nelements = length(elset_ids)
        element_ids = get(elements, "NUM", collect(1:nelements))
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

"""
    aster_read_mesh(filename, mesh_name=nothing)

Parse code aster .med file and return mesh data in a dictionary.

Dictionary contains additional dictionaries `nodes`, `node_sets`, `elements`,
`element_sets`, `element_types`, `surface_sets` and `surface_types`.

If mesh file contains several meshes, one must provide the mesh name as
additional argument or expcetion will be thrown.

"""
function aster_read_mesh(filename::String, mesh_name=nothing)
    isfile(filename) || error("Cannot read mesh file $filename: file not found.")
    aster_read_mesh_(MEDFile(filename), mesh_name)
end

function aster_read_mesh_(med::MEDFile, mesh_name=nothing)
    mesh_names = get_mesh_names(med)
    all_meshes = join(mesh_names, ", ")
    if mesh_name == nothing
        length(mesh_names) == 1 || error("several meshes found from med, pick one: $all_meshes")
        mesh_name = mesh_names[1]
    else
        mesh_name in mesh_names || error("Mesh $mesh_name not found from mesh file $fn. Available meshes: $all_meshes")
    end

    mesh = Dict{String, Dict}()
    mesh["nodes"] = Dict{Int, Vector{Float64}}()
    mesh["node_sets"] = Dict{String, Vector{Int}}()
    mesh["elements"] = Dict{Int, Vector{Int}}()
    mesh["element_types"] = Dict{Int, Symbol}()
    mesh["element_sets"] = Dict{String, Vector{Int}}()
    mesh["surface_sets"] = Dict{String, Vector{Tuple{Int, Symbol}}}()
    mesh["surface_types"] = Dict{String, Symbol}()

    elsets = get_element_sets(med, mesh_name)
    nsets = get_node_sets(med, mesh_name)
    for (nid, (nset_, coords)) in get_nodes(med, nsets, mesh_name)
        mesh["nodes"][nid] = coords
        for nset in nset_
            if !haskey(mesh["node_sets"], nset)
                mesh["node_sets"][nset] = []
            end
            push!(mesh["node_sets"][nset], nid)
        end
    end
    for (elid, (eltyp, elset_, elcon)) in get_connectivity(med, elsets, mesh_name)
        mesh["elements"][elid] = elcon
        mesh["element_types"][elid] = eltyp
        for elset in elset_
            if !haskey(mesh["element_sets"], elset)
                mesh["element_sets"][elset] = []
            end
            push!(mesh["element_sets"][elset], elid)
        end
    end
    return mesh
end
