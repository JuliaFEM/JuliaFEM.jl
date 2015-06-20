# This file is a part of JuliaFEM. License is MIT: https://github.com/ovainola/JuliaFEM/blob/master/README.md
module JuliaFEM

export Model, new_model, new_field, get_field


type Model
    model  # For global variables
    nodes  # For nodes
    elements  # For elements
    element_nodes  # For element nodes
    element_gauss_points  # For element gauss points
end

function new_model()
    """Initialize empty model.
    """
    return Model(Dict(), Dict(), Dict(), Dict(), Dict())
end

function new_field(field_type, field_name)
    """Create new field to model.

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

    """
    d = Dict()
    setindex!(field_type, d, field_name)
    return d
end

function get_field(field_type, field_name; create_if_doesnt_exist=false)
    """Get field from model.

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
    """
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



end # module
