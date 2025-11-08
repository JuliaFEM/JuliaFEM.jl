# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/FEMQuad.jl/blob/master/LICENSE

### Gauss quadrature rules for triangular elements

""" Gauss-Legendre quadrature, 1 point rule on triangle. """
function get_quadrature_points(::Type{Val{:GLTRI1}})
    weights = (0.5, )
    points = ((1.0/3.0, 1.0/3.0), )
    return zip(weights, points)
end

function get_order(::Type{Val{:GLTRI1}})
    return 1
end

""" Gauss-Legendre quadrature, 3 point rule on triangle. """
function get_quadrature_points(::Type{Val{:GLTRI3}})
    weights = (1.0/6.0, 1.0/6.0, 1.0/6.0)
    points = (
              (2.0/3.0, 1.0/6.0),
              (1.0/6.0, 2.0/3.0),
              (1.0/6.0, 1.0/6.0)
             )
    return zip(weights, points)
end

function get_order(::Type{Val{:GLTRI3}})
    return 2
end

""" Gauss-Legendre quadrature, 3 point rule on triangle. """
function get_quadrature_points(::Type{Val{:GLTRI3B}})
    weights = (1.0/6.0, 1.0/6.0, 1.0/6.0)
    points = (
              (0.0, 1.0/2.0),
              (1.0/2.0, 0.0),
              (1.0/2.0, 1.0/2.0)
             )
    return zip(weights, points)
end

function get_order(::Type{Val{:GLTRI3B}})
    return 2
end

""" Gauss-Legendre quadrature, 4 point rule on triangle. """
function get_quadrature_points(::Type{Val{:GLTRI4}})
    weights = (
        1.5902069087198858469718450103758e-01,
        9.0979309128011415302815498962418e-02,
        1.5902069087198858469718450103758e-01,
        9.0979309128011415302815498962418e-02)
    points = (
        (1.5505102572168219018027159252941e-01, 1.7855872826361642311703513337422e-01),
        (6.4494897427831780981972840747059e-01, 7.5031110222608118177475598324603e-02),
        (1.5505102572168219018027159252941e-01, 6.6639024601470138670269327409637e-01),
        (6.4494897427831780981972840747059e-01, 2.8001991549907407200279599420481e-01))
    return zip(weights, points)
end

function get_order(::Type{Val{:GLTRI4}})
    return 3
end

""" Gauss-Legendre quadrature, 4 point rule on triangle. """
function get_quadrature_points(::Type{Val{:GLTRI4B}})
    weights = (-27.0/96.0, 25.0/96.0, 25.0/96.0, 25.0/96.0)
    points = (
              (1.0/3.0, 1.0/3.0),
              (1.0/5.0, 1.0/5.0),
              (1.0/5.0, 3.0/5.0),
              (3.0/5.0, 1.0/5.0))
    return zip(weights, points)
end

function get_order(::Type{Val{:GLTRI4B}})
    return 3
end

""" Gauss-Legendre quadrature, 6 point rule on triangle. """
function get_quadrature_points(::Type{Val{:GLTRI6}})
    P1 = 0.11169079483905
    P2 = 0.0549758718227661
    A = 0.445948490915965
    B = 0.091576213509771
    weights = (P2, P2, P2, P1, P1, P1)
    points = ((B, B), (1.0-2.0*B, B), (B, 1.0-2.0*B),
              (A, 1.0-2*A), (A, A), (1.0 - 2.0*A, A))
    return zip(weights, points)
end

function get_order(::Type{Val{:GLTRI6}})
    return 4
end

""" Gauss-Legendre quadrature, 7 point rule on triangle. """
function get_quadrature_points(::Type{Val{:GLTRI7}})
    A = 0.470142064105115
    B = 0.101286507323456
    P1 = 0.066197076394253
    P2 = 0.062969590272413
    weights = (9/80, P1, P1, P1, P2, P2, P2)
    points = ((1/3, 1/3), (A, A), (1.0-2.0*A, A), (A, 1.0-2.0*A),
              (B, B), (1.0-2.0*B, B), (B, 1.0-2.0*B))
    return zip(weights, points)
end

function get_order(::Type{Val{:GLTRI7}})
    return 5
end

""" Gauss-Legendre quadrature, 12 point rule on triangle. """
function get_quadrature_points(::Type{Val{:GLTRI12}})
    A = 0.063089014491502
    B = 0.249286745170910
    C = 0.310352451033785
    D = 0.053145049844816
    P1 = 0.025422453185103
    P2 = 0.058393137863189
    P3 = 0.041425537809187
    weights = (P1, P1, P1, P2, P2, P2, P3, P3, P3, P3, P3, P3)
    points = ((A, A), (1.0-2.0*A, A), (A, 1.0-2.0*A), (B, B), (1.0-2.0*B, B),
              (B, 1.0-2.0*B), (C, D), (D, C), (1.0-C-D, C), (1.0-C-D, D),
              (C, 1.0-C-D), (D, 1.0-C-D))
    return zip(weights, points)
end

function get_order(::Type{Val{:GLTRI12}})
    return 6
end
