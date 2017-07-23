# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using FEMQuad: get_quadrature_points

# default number of integration points for each element
# first rule is the default
# some times we want to increase integration order, e.g. when integrating mass
# matrix or boundary conditions, so we add additional scheme to do so and that's
# the second rule
integration_rule_mapping = (
    :Seg2    => (:GLSEG1,  :GLSEG2),
    :Seg3    => (:GLSEG2,  :GLSEG3),
    :NSeg    => (:GLSEG2,  :GLSEG3),
    :Quad4   => (:GLQUAD1, :GLQUAD4),
    :Quad8   => (:GLQUAD4, :GLQUAD9),
    :Quad9   => (:GLQUAD4, :GLQUAD9),
    :NSurf   => (:GLQUAD4, :GLQUAD9),
    :Hex8    => (:GLHEX1,  :GLHEX27),
    :Hex20   => (:GLHEX27, :GLHEX81),
    :Hex27   => (:GLHEX27, :GLHEX81),
    :NSolid  => (:GLHEX27, :GLHEX81),
    :Tri3    => (:GLTRI1,  :GLTRI3),
    :Tri6    => (:GLTRI3,  :GLTRI4),
    :Tri7    => (:GLTRI3,  :GLTRI4),
    :Tet4    => (:GLTET1,  :GTET4),
    :Tet10   => (:GLTET4,  :GLTET5),
    :Pyr5    => (:GLPYR5,  ),
    :Wedge6  => (:GLWED6,  :GLWED21),
    :Wedge15 => (:GLWED21, ))

for (E, R) in integration_rule_mapping
    P = Val{first(R)}
    code = quote
        function get_integration_points(element::$E)
            return get_quadrature_points($P)
        end
    end
    eval(code)
    if length(R) > 1
        P = Val{(R[2])}
        code = quote
            function get_integration_points(element::$E, ::Type{Val{1}})
                return get_quadrature_points($P)
            end
        end
        eval(code)
    end
end
