# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Gauss-Legendre quadrature rules for pyramid elements.

This file implements quadrature rules for 3D pyramid reference elements
with a quadrilateral base and triangular apex.

# Available Rules
- `GaussLegendre{2}()`: 5-point rule (default variant), exact for linear
- `GaussLegendre{2,:B}()`: 5-point alternative variant, exact for linear

# Notes
- Pyramid has quadrilateral base at z=0 and apex at (0,0,1)
- Quadrature for pyramids is non-trivial due to singularity at apex
- Both variants have order 1 accuracy
- Higher order rules for pyramids are complex and rarely implemented

See also: [`GaussLegendre`](@ref), [`QuadraturePoint`](@ref)
"""

# ============================================================================
# GaussLegendre{2}: 5-point rules (two variants)
# ============================================================================

"""
    get_quadrature_points(::Type{Pyramid}, ::GaussLegendre{2})

5-point Gauss-Legendre rule for pyramid (default variant).
Exact for linear polynomials (degree 1).

4 points on base plane + 1 point elevated toward apex.
"""
@inline function get_quadrature_points(::Type{Pyramid}, ::GaussLegendre{2,:default})
    g1 = 0.5842373946721771876874344
    g2 = -2/3
    g3 = 2/5
    w1 = 81/100
    w2 = 125/27

    return SVector(
        QuadraturePoint(Vec{3}(-g1, -g1, g2), w1),
        QuadraturePoint(Vec{3}( g1, -g1, g2), w1),
        QuadraturePoint(Vec{3}( g1,  g1, g2), w1),
        QuadraturePoint(Vec{3}(-g1,  g1, g2), w1),
        QuadraturePoint(Vec{3}(0.0, 0.0, g3), w2)
    )
end

"""
    get_quadrature_points(::Type{Pyramid}, ::GaussLegendre{2,:B})

5-point Gauss-Legendre rule for pyramid (variant B).
Exact for linear polynomials (degree 1).

Alternative point distribution with uniform weights.
"""
@inline function get_quadrature_points(::Type{Pyramid}, ::GaussLegendre{2,:B})
    a = 2/15
    h1 = 0.1531754163448146
    h2 = 0.6372983346207416

    return SVector(
        QuadraturePoint(Vec{3}( 0.5,  0.0, h1), a),
        QuadraturePoint(Vec{3}( 0.0,  0.5, h1), a),
        QuadraturePoint(Vec{3}(-0.5,  0.0, h1), a),
        QuadraturePoint(Vec{3}( 0.0, -0.5, h1), a),
        QuadraturePoint(Vec{3}( 0.0,  0.0, h2), a)
    )
end

# ============================================================================
# Legacy symbol-based API (deprecated, kept for backwards compatibility)
# ============================================================================

# These map old :GLPYR symbols to new API
get_quadrature_points(::Type{Val{:GLPYR5}}) =
    get_quadrature_points(Pyramid, GaussLegendre{2}())

get_quadrature_points(::Type{Val{:GLPYR5B}}) =
    get_quadrature_points(Pyramid, GaussLegendre{2,:B}())

# Order queries (deprecated)
get_order(::Type{Val{:GLPYR5}}) = 1
get_order(::Type{Val{:GLPYR5B}}) = 1
