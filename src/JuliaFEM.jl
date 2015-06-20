# This file is a part of JuliaFEM. License is MIT: https://github.com/ovainola/JuliaFEM/blob/master/README.md
module JuliaFEM

export Model, new_model

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

end # module
