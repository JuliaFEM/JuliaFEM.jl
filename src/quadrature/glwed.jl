# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/FEMQuad.jl/blob/master/LICENSE

### Gauss quadrature rules for prismatic elements (wedge)

""" Gauss-Legendre quadrature, 6 point rule on wedge. """
function get_quadrature_points(::Type{Val{:GLWED6}})
    w = 1.0/6.0
    a = sqrt(1.0/3.0)
    weights = (w, w, w, w, w, w)
    points = ((0.5, 0.0, -a), (0.0, 0.5, -a), (0.5, 0.5, -a),
              (0.5, 0.0,  a), (0.0, 0.5,  a), (0.5, 0.5,  a))
    return zip(weights, points)
end

function get_order(::Type{Val{:GLWED6}})
    return 1
end

""" Gauss-Legendre quadrature, 6 point rule on wedge. """
function get_quadrature_points(::Type{Val{:GLWED6B}})
    w = 1.0/6.0
    a = sqrt(1.0/3.0)
    weights = (w, w, w, w, w, w)
    points = ((2.0/3.0, 1.0/6.0, -a), (1.0/6.0, 2.0/3.0, -a), (1.0/6.0, 1.0/6.0, -a),
              (2.0/3.0, 1.0/6.0,  a), (1.0/6.0, 2.0/3.0,  a), (1.0/6.0, 1.0/6.0,  a))
    return zip(weights, points)
end

function get_order(::Type{Val{:GLWED6B}})
    return 1
end

""" Gauss-Legendre quadrature, 21 point rule on wedge. """
function get_quadrature_points(::Type{Val{:GLWED21}})
    alpha = sqrt(3/5)
    c1 = 5/9
    c2 = 8/9
    a = (6+sqrt(15))/21
    b = (6-sqrt(15))/21

    weights = (
        c1*9/80,
        c1*((155+sqrt(15))/2400),
        c1*((155+sqrt(15))/2400),
        c1*((155+sqrt(15))/2400),
        c1*((155-sqrt(15))/2400),
        c1*((155-sqrt(15))/2400),
        c1*((155-sqrt(15))/2400),
        c2*9/80,
        c2*((155+sqrt(15))/2400),
        c2*((155+sqrt(15))/2400),
        c2*((155+sqrt(15))/2400),
        c2*((155-sqrt(15))/2400),
        c2*((155-sqrt(15))/2400),
        c2*((155-sqrt(15))/2400),
        c1*9/80,
        c1*((155+sqrt(15))/2400),
        c1*((155+sqrt(15))/2400),
        c1*((155+sqrt(15))/2400),
        c1*((155-sqrt(15))/2400),
        c1*((155-sqrt(15))/2400),
        c1*((155-sqrt(15))/2400))

    points = (
          (1/3, 1/3, -alpha),
          (a, a, -alpha),
          (1-2a, a, -alpha),
          (a, 1-2a, -alpha),
          (b, b, -alpha),
          (1-2b, b, -alpha),
          (b, 1-2b, -alpha),
          (1/3, 1/3, 0),
          (a, a, 0),
          (1-2a, a, 0),
          (a, 1-2a, 0),
          (b, b, 0),
          (1-2b, b, 0),
          (b, 1-2b, 0),
          (1/3, 1/3, alpha),
          (a, a, alpha),
          (1-2a, a, alpha),
          (a, 1-2a, alpha),
          (b, b, alpha),
          (1-2b, b, alpha),
          (b, 1-2b, alpha))

    return zip(weights, points)
end

function get_order(::Type{Val{:GLWED21}})
    return 5
end
