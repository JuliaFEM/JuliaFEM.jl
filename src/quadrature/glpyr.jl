# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/FEMQuad.jl/blob/master/LICENSE

### Gauss quadrature rules for pyramid elements

""" Gauss-Legendre quadrature, 5 point rule on pyramid. """
function get_quadrature_points(::Type{Val{:GLPYR5}})
    g1 =  0.5842373946721771876874344
    g2 = -2.0/3.0
    g3 =  2.0/5.0
    w1 =  81.0/100.0
    w2 =  125.0/27.0
    weights = (w1, w1, w1, w1, w2)
    points = (
              (-g1, -g1, g2),
              ( g1, -g1, g2),
              ( g1,  g1, g2),
              (-g1,  g1, g2),
              (0.0, 0.0, g3))
    return zip(weights, points)
end

function get_order(::Type{Val{:GLPYR5}})
    return 1
end

""" Gauss-Legendre quadrature, 5 point rule on pyramid. """
function get_quadrature_points(::Type{Val{:GLPYR5B}})
    a = 2.0/15.0
    h1 = 0.1531754163448146
    h2 = 0.6372983346207416
    weights = (a, a, a, a, a)
    points = (
              (0.5, 0.0, h1),
              (0.0, 0.5, h1),
              (-0.5, 0.0, h1),
              (0.0, -0.5, h1),
              (0.0, 0.0, h2)
             )
    return zip(weights, points)
end

function get_order(::Type{Val{:GLPYR5B}})
    return 1
end
