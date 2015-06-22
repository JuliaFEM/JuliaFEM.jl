using FactCheck
#using JuliaFEM
using Logging
using HDF5
@Logging.configure(level=DEBUG)

@doc """
Create new field to model.

Parameters
----------
field_type : Dict()
    Target topology (model.model, model.nodes, model.elements,
                     model.element_nodes, model.element_gauss
field_name : str
    Field name

Returns
-------
Dict
    New field

""" ->
function new_field!(model, field_type, field_name; partition=1, time=0, increment=0)
  h5write("$model.$partition.h5", "$time/$increment/$field_type/$field_name", Float64[])
end


@doc """Add new nodes to model.

Parameters
----------
model : str
  Path to model file
nodes : Dict()
  id => coords
partition : int, optional
  Partition number, defaults to 1
time : float, optional
  Time step, defaults to 0
increment : float, optional
  Increment number, defaults to 0

Returns
-------
None

Notes
-----
Create new field "coords" to model if not found
""" ->
function add_nodes!(model, nodes; partition=1, time=0, increment=0)
  coords = Float64[]
  node_ids = Int64[]
  for (nid, ncoords) in nodes
    @debug("adding nid: ", nid, " with coords: ", ncoords)
    @assert length(ncoords) == 3
    for x in ncoords
      push!(coords, x)
    end
    push!(node_ids, nid)
  end
  h5write("$model.$partition.h5", "$time/$increment/nodes/coords", coords)
  h5write("$model.$partition.h5", "$time/$increment/nodes/node_ids", node_ids)
end



@doc """Return subset of nodes from model.

Parameters
----------
node_ids : array, optional
  List of node ids to return. If not given, return all nodes.
partition : int, optional
  Partition number, defaults to 1
time : float, optional
  Time step, defaults to 0
increment : float, optional
  Increment number, defaults to 0

Returns
-------
Dict()
    id => coords
""" ->
function get_nodes(model, node_ids=[]; partition=1, time=0, increment=0)
  subset = Dict()
  all_node_coords = h5read("$model.1.h5", "$time/$increment/nodes/coords")
  all_node_ids = h5read("$model.1.h5", "$time/$increment/nodes/node_ids")
  @debug("all node coords: ", all_node_coords)
  @debug("all node ids: ", all_node_ids)
  if length(node_ids) == 0
    node_ids = all_node_ids
  end
  dim = 3
  for node_id in node_ids
    @debug("fetching node ", node_id)
    idx = findfirst(all_node_ids, node_id)
    @debug("found node coords from idx ", idx)
    subset[node_id] = all_node_coords[dim*(idx-1)+1:dim*(idx-1)+dim]
  end
  return subset
end



facts("create new field to model") do
  model = tempname()
  @debug("model file name", model)
  new_field!(model, "nodes", "coords")
  data = h5read("$model.1.h5", "0/0/nodes/coords")
  @fact length(data) => 0
end

facts("add nodes to model") do
  model = tempname()
  nodes = Dict(1 => [1.0, 2.0, 3.0])
  add_nodes!(model, nodes)
  @fact h5read("$model.1.h5", "0/0/nodes/coords") => [1.0, 2.0, 3.0]
  @fact h5read("$model.1.h5", "0/0/nodes/node_ids") => [1]
end

facts("get nodes from model") do
  model = tempname()
  nodes = Dict(1 => [1.0, 2.0, 3.0], 2 => [2.0, 3.0, 4.0])
  add_nodes!(model, nodes)
  subset = get_nodes(model, [1])
  @fact subset[1] => [1.0, 2.0, 3.0]
  subset = get_nodes(model, [2])
  @fact subset[2] => [2.0, 3.0, 4.0]
  subset = get_nodes(model, [1,2])
  @fact subset[1] => [1.0, 2.0, 3.0]
  @fact subset[2] => [2.0, 3.0, 4.0]
end