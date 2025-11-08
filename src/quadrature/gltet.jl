# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/FEMQuad.jl/blob/master/LICENSE

### Gauss quadrature rules for tetrahedrons

""" Gauss-Legendre quadrature, 1 point rule on tetrahedron. """
function get_quadrature_points(::Type{Val{:GLTET1}})
    weights = (1.0/6.0, )
    points = ((1.0/4.0, 1.0/4.0, 1.0/4.0), )
    return zip(weights, points)
end

function get_order(::Type{Val{:GLTET1}})
    return 1
end

""" Gauss-Legendre quadrature, 4 point rule on tetrahedron. """
function get_quadrature_points(::Type{Val{:GLTET4}})
    a = (5.0+3.0*sqrt(5.0))/20.0
    b = (5.0-sqrt(5.0))/20.0
    w = 1.0/24.0
    weights = (w, w, w, w)
    points = ((a, b, b), (b, a, b), (b, b, a), (b, b, b))
    return zip(weights, points)
end

function get_order(::Type{Val{:GLTET4}})
    return 2
end

""" Gauss-Legendre quadrature, 5 point rule on tetrahedron. """
function get_quadrature_points(::Type{Val{:GLTET5}})
    a = 1.0/4.0
    b = 1.0/6.0
    c = 1.0/2.0
    weights = (-2.0/15.0, 3.0/40.0, 3.0/40.0, 3.0/40.0, 3.0/40.0)
    points = ((a, a, a), (b, b, b), (b, b, c), (b, c, b), (c, b, b))
    return zip(weights, points)
end

function get_order(::Type{Val{:GLTET5}})
    return 3
end

""" Gauss-Legendre quadrature, 15 point rule on tetrahedron. """
function get_quadrature_points(::Type{Val{:GLTET15}})
    a = 1.0/4.0
    b1 = 1.0/34.0*(7.0 + sqrt(15.0))
    b2 = 1.0/34.0*(7.0 - sqrt(15.0))
    c1 = 1.0/34.0*(13.0 - 3.0*sqrt(15.0))
    c2 = 1.0/34.0*(13.0 + 3.0*sqrt(15.0))
    d = 1.0/20.0*(5.0 - sqrt(15.0))
    f = 1.0/20.0*(5.0 + sqrt(15.0))
    w1 = 8.0/405.0
    w2 = (2665.0 - 14.0*sqrt(15.0))/226800.0
    w3 = (2665.0 + 14.0*sqrt(15.0))/226800.0
    w4 = 5.0/567.0
    weights = (w1, w2, w2, w2, w2, w3, w3, w3, w3, w4, w4, w4, w4, w4, w4)
    points = ((a, a, a), (b1, b1, b1), (b1, b1, c1), (b1, c1, b1), (c1, b1, b1),
              (b2, b2, b2), (b2, b2, c2), (b2, c2, b2), (c2, b2, b2), (d, d, f),
              (d, f, d), (f, d, d), (d, f, f), (f, d, f), (f, f, d))
    return zip(weights, points)
end

function get_order(::Type{Val{:GLTET15}})
    return 4
end
