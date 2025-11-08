# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/AsterReader.jl/blob/master/LICENSE

module AsterReader

using HDF5

"""
    parse_node_id(node_name)

Return id number from node name. Usually the node name contains the id number,
i.e. N123 => 123 and so on.

"""
function parse_node_id(node_name)
    m = match(r"\d+", node_name)
    m != nothing || error("Unable to parse id from node name $node_name.")
    return tryparse(Int, m.match)
end

include("read_aster_mesh.jl")
include("read_aster_results.jl")
export aster_read_mesh

end
