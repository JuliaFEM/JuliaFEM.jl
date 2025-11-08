# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/FEMBase.jl/blob/master/LICENSE

# Default number of integration points for each element. First rule is the
# default integration rule returned by `get_integration_points(element)`.
# Sometimes we want to increase integration order, e.g. when integrating mass
# matrix or boundary conditions. For that reason, additional rules are provied
# in list, so e.g. `get_integration_points(element, 1)` returns the second rule,
# `get_integration_points(element, 2)` third rule and so on. Rules should be
# ordered so that picking next one integrates more accurately.
integration_rule_mapping = (
    :Seg2 => (:GLSEG2, :GLSEG3, :GLSEG4, :GLSEG5),
    :Seg3 => (:GLSEG3, :GLSEG4, :GLSEG5),
    :NSeg => (:GLSEG2, :GLSEG3, :GLSEG4, :GLSEG5),
    :Quad4 => (:GLQUAD4, :GLQUAD9, :GLQUAD16, :GLQUAD25),
    :Quad8 => (:GLQUAD9, :GLQUAD16, :GLQUAD25),
    :Quad9 => (:GLQUAD9, :GLQUAD16, :GLQUAD25),
    :NSurf => (:GLQUAD9, :GLQUAD16, :GLQUAD25),
    :Hex8 => (:GLHEX8, :GLHEX27, :GLHEX64, :GLHEX125),
    :Hex20 => (:GLHEX27, :GLHEX64, :GLHEX125),
    :Hex27 => (:GLHEX27, :GLHEX64, :GLHEX125),
    :NSolid => (:GLHEX27, :GLHEX64, :GLHEX125),
    :Tri3 => (:GLTRI1, :GLTRI3, :GLTRI4, :GLTRI6, :GLTRI7, :GLTRI12),
    :Tri6 => (:GLTRI3, :GLTRI4, :GLTRI6, :GLTRI7, :GLTRI12),
    :Tri7 => (:GLTRI3, :GLTRI4, :GLTRI6, :GLTRI7, :GLTRI12),
    :Tet4 => (:GLTET1, :GLTET4, :GLTET5, :GLTET15),
    :Tet10 => (:GLTET4, :GLTET5, :GLTET15),
    :Pyr5 => (:GLPYR5,),
    :Wedge6 => (:GLWED6, :GLWED21),
    :Wedge15 => (:GLWED21,))

for (E, R) in integration_rule_mapping
    for i in 1:length(R)
        P = Val{R[i]}
        order = Val{i - 1}
        local code  # Explicitly declare as local to avoid warning
        if isequal(i, 1)
            code = quote
                function get_integration_points(element::$E)
                    return get_quadrature_points($P)
                end
            end
        else
            code = quote
                function get_integration_points(element::$E, ::Type{$order})
                    return get_quadrature_points($P)
                end
            end
        end
        eval(code)
    end
end

# All good codes needs a special case. Here we have it: Poi1
function get_integration_points(::Poi1)
    [(1.0, (0.0,))]
end
