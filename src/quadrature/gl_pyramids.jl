# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Gauss-Legendre quadrature rules for pyramid elements.
"""

# ============================================================================
# GaussLegendre{2}: 5-point rules (two variants)
# ============================================================================

"""5-point Gauss-Legendre rule for pyramid (default variant)."""
@inline function get_quadrature_points(::Type{Pyramid}, ::GaussLegendre{2,:default})
    g1 = 0.5842373946721771876874344
    g2 = -2/3
    g3 = 2/5
    w1 = 81/100
    w2 = 125/27

    return SVector(
        QuadraturePoint(Vec{3}((-g1, -g1, g2)), w1),
        QuadraturePoint(Vec{3}(( g1, -g1, g2)), w1),
        QuadraturePoint(Vec{3}(( g1, g1, g2)), w1),
        QuadraturePoint(Vec{3}((-g1, g1, g2)), w1),
        QuadraturePoint(Vec{3}((0.0, 0.0, g3)), w2)
    )
end

"""5-point Gauss-Legendre rule for pyramid (variant B)."""
@inline function get_quadrature_points(::Type{Pyramid}, ::GaussLegendre{2,:B})
    a = 2/15
    h1 = 0.1531754163448146
    h2 = 0.6372983346207416

    return SVector(
        QuadraturePoint(Vec{3}(( 0.5, 0.0, h1)), a),
        QuadraturePoint(Vec{3}(( 0.0, 0.5, h1)), a),
        QuadraturePoint(Vec{3}((-0.5, 0.0, h1)), a),
        QuadraturePoint(Vec{3}(( 0.0, -0.5, h1)), a),
        QuadraturePoint(Vec{3}(( 0.0, 0.0, h2)), a)
    )
end

