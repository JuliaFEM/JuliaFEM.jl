# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/AsterReader.jl/blob/master/LICENSE

""" Code Aster result file (.rmed). """
mutable struct RMEDFile
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
    node_ids = map(parse_node_id, node_names)
    nodes = Dict(j => node_coords[:,j] for j in node_ids)
    return nodes
end

""" Read nodal field from rmed file. """
function aster_read_data(rmed::RMEDFile, field_name; field_type=:NODE,
                         info_fields=true, node_ids=nothing)

    if occursin("ELGA", field_name)
        field_type = :GAUSS
    end

    if node_ids == nothing
        nodes = aster_read_nodes(rmed)
        node_ids = sort(collect(keys(nodes)))
    end

    if info_fields
        field_names = keys(rmed.data["CHA"])
        all_fields = join(field_names, ", ")
        @info("results: $all_fields")
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
