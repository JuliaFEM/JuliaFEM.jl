# This file is a part of JuliaFEM. License is MIT: https://github.com/ovainola/JuliaFEM/blob/master/README.md
module JuliaFEM

export Model, new_model, new_field, get_field, add_nodes, get_nodes, add_elements, get_elements


@doc """
Basic model
""" ->
type Model
    model  # For global variables
    nodes  # For nodes
    elements  # For elements
    element_nodes  # For element nodes
    element_gauss_points  # For element gauss points
end




@doc """
Initialize empty model.

Parameters
----------
None

Returns
-------
New model struct
""" ->
function new_model()
    return Model(Dict(), Dict(), Dict(), Dict(), Dict())
end




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
function new_field(field_type, field_name)
    d = Dict()
    setindex!(field_type, d, field_name)
    return d
end




@doc """Get field from model.

Parameters
----------
field_type : Dict()
    Target topology (model.model, model.nodes, model.elements,
                     model.element_nodes, model.element_gauss
field_name : str
    Field name

create_if_doesnt_exist : bool, optional
    If field doesn't exists, create one and return empty field

Raises
------
Error, if field not found and create_if_doesnt_exist == false
""" ->
function get_field(field_type, field_name; create_if_doesnt_exist=false)
    if !(field_name in keys(field_type))
        if create_if_doesnt_exist
            field = new_field(field_type, field_name)
        else
            throw("Field not found")
        end
    else
        field = getindex(field_type, field_name)
    end
    return field
end




@doc """Add new nodes to model.

Parameters
----------
nodes : Dict()
    id => coords

Returns
-------
model

Notes
-----
Create new field "coords" to model if not found
""" ->
function add_nodes(model, nodes)
    #pri("Adding ", length(nodes), " nodes to model")
    field = get_field(model.nodes, "coords"; create_if_doesnt_exist=true)
    for (node_id, coords) in nodes
        field[node_id] = coords
    end
end




@doc """Return subset of nodes from model.

Parameters
----------
node_ids : array
    list of node ids to return

Returns
-------
Dict()
    id => coords
""" ->
function get_nodes(model, node_ids)
    subset = Dict()
    for node_id in node_ids
        subset[node_id] = model.nodes["coords"][node_id]
    end
    return subset
end




@doc """Add new elements to model.
Parameters
----------
list of dicts, dict = {element_type => elcode, elids => [node ids..]}

Examples
--------
Create two tet4 element and add them:

>>> m = new_model()
>>> const TET4 = 0x6
>>> el1 = Dict("element_type" => TET4, "node_ids" => [1, 2, 3, 4])
>>> el2 = Dict("element_type" => TET4, "node_ids" => [4, 3, 2, 1])

In dict key means element id

>>> elements = Dict(1 => el1, 2 => el2)
>>> add_elements(m, elements)

""" ->
function add_elements(model, elements)
    elfield = get_field(model.elements, "connectivity"; create_if_doesnt_exist=true)
    eltyfield = get_field(model.elements, "element_type"; create_if_doesnt_exist=true)
    for (elid, element) in elements
        #prin*tln("Adding element ", elid)
        elfield[elid] = element["node_ids"]
        eltyfield[elid] = element["element_type"]
    end
end




@doc """Get subset of elements from model.
Parameters
----------
element_ids : list of ints
    Element id numbers

Returns
-------
Dict
{element_type = XXX, node_ids = [a, b, c, d, e, ..., n]}
""" ->
function get_elements(model, element_ids)
    eltyfield = get_field(model.elements, "element_type")
    elfield = get_field(model.elements, "connectivity")
    ret = Dict()
    for i in element_ids
        ret[i] = Dict("element_type" => eltyfield[i], "node_ids" => elfield[i])
    end
    return ret
end



end # module
