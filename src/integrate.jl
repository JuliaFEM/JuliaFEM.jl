# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using FEMQuad: get_quadrature_points

# default number of integration points for each element
get_integration_points(element::Seg2) = get_quadrature_points(::Type{Val{:GLSEG1}})
get_integration_points(element::Seg3) = get_quadrature_points(::Type{Val{:GLSEG2}})
get_integration_points(element::NSeg) = get_quadrature_points(::Type{Val{:GLSEG2}})
get_integration_points(element::Quad4) = get_quadrature_points(::Type{Val{:GLQUAD1}})
get_integration_points(element::Quad8) = get_quadrature_points(::Type{Val{:GLQUAD4}})
get_integration_points(element::Quad9) = get_quadrature_points(::Type{Val{:GLQUAD4}})
get_integration_points(element::NSurf) = get_quadrature_points(::Type{Val{:GLQUAD4}})
get_integration_points(element::Hex8) = get_quadrature_points(::Type{Val{:GLHEX1}})
get_integration_points(element::Hex20) = get_quadrature_points(::Type{Val{:GLHEX27}})
get_integration_points(element::Hex27) = get_quadrature_points(::Type{Val{:GLHEX27}})
get_integration_points(element::NSolid) = get_quadrature_points(::Type{Val{:GLHEX27}})
get_integration_points(element::Tri3) = get_quadrature_points(::Type{Val{:GLTRI1}})
get_integration_points(element::Tri6) = get_quadrature_points(::Type{Val{:GLTRI3}})
get_integration_points(element::Tri7) = get_quadrature_points(::Type{Val{:GLTRI3}})
get_integration_points(element::Tet4) = get_quadrature_points(::Type{Val{:GLTET1}})
get_integration_points(element::Tet10) = get_quadrature_points(::Type{Val{:GLTET4}})
get_integration_points(element::Pyr5) = get_quadrature_points(::Type{Val{:GLPYR5B}})
get_integration_points(element::Wedge6) = get_quadrature_points(::Type{Val{:GLWED6}})
get_integration_points(element::Wedge15) = get_quadrature_points(::Type{Val{:GLWED21}})
