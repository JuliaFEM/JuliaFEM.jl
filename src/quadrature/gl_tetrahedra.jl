# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Gauss-Legendre quadrature rules for tetrahedral elements.

This file implements quadrature rules for 3D tetrahedral reference elements
in the parametric domain with vertices at (0,0,0), (1,0,0), (0,1,0), (0,0,1).

# Available Rules
- `GaussLegendre{1}()`: 1-point rule (centroid), exact for linear
- `GaussLegendre{2}()`: 4-point rule, exact for quadratic
- `GaussLegendre{3}()`: 5-point rule, exact for cubic (has negative weight!)
- `GaussLegendre{4}()`: 15-point rule, exact for quartic

# Notes
- Reference tetrahedron has constraint ξ + η + ζ ≤ 1
- All weights sum to 1/6 (volume of reference tetrahedron)
- GLTET5 has one negative weight, may cause numerical issues
- Higher order rules (>4) are available in literature but rarely used

See also: [`GaussLegendre`](@ref), [`QuadraturePoint`](@ref)
"""

# ============================================================================
# GaussLegendre{1}: 1-point rule (centroid)
# ============================================================================

"""
    get_quadrature_points(::Type{Tetrahedron}, ::GaussLegendre{1})

1-point Gauss-Legendre rule for tetrahedron (centroid).
Exact for linear polynomials (degree 1).
"""
@inline function get_quadrature_points(::Type{Tetrahedron}, ::GaussLegendre{1,V}) where V
    return SVector(
        QuadraturePoint(Vec{3}(1 / 4, 1 / 4, 1 / 4), 1 / 6)
    )
end

# ============================================================================
# GaussLegendre{2}: 4-point rule
# ============================================================================

"""
    get_quadrature_points(::Type{Tetrahedron}, ::GaussLegendre{2})

4-point Gauss-Legendre rule for tetrahedron.
Exact for quadratic polynomials (degree 2).

Points are symmetrically placed inside the tetrahedron.
"""
@inline function get_quadrature_points(::Type{Tetrahedron}, ::GaussLegendre{2,V}) where V
    a = (5 + 3 * sqrt(5)) / 20
    b = (5 - sqrt(5)) / 20
    w = 1 / 24

    return SVector(
        QuadraturePoint(Vec{3}(a, b, b), w),
        QuadraturePoint(Vec{3}(b, a, b), w),
        QuadraturePoint(Vec{3}(b, b, a), w),
        QuadraturePoint(Vec{3}(b, b, b), w)
    )
end

# ============================================================================
# GaussLegendre{3}: 5-point rule
# ============================================================================

"""
    get_quadrature_points(::Type{Tetrahedron}, ::GaussLegendre{3})

5-point Gauss-Legendre rule for tetrahedron.
Exact for cubic polynomials (degree 3).

**WARNING:** This rule has one negative weight (-2/15), which may cause
numerical issues. Consider using GaussLegendre{2}() or {4}() instead.
"""
@inline function get_quadrature_points(::Type{Tetrahedron}, ::GaussLegendre{3,V}) where V
    a = 1 / 4
    b = 1 / 6
    c = 1 / 2

    return SVector(
        QuadraturePoint(Vec{3}(a, a, a), -2 / 15),  # Negative weight!
        QuadraturePoint(Vec{3}(b, b, b), 3 / 40),
        QuadraturePoint(Vec{3}(b, b, c), 3 / 40),
        QuadraturePoint(Vec{3}(b, c, b), 3 / 40),
        QuadraturePoint(Vec{3}(c, b, b), 3 / 40)
    )
end

# ============================================================================
# GaussLegendre{4}: 15-point rule
# ============================================================================

"""
    get_quadrature_points(::Type{Tetrahedron}, ::GaussLegendre{4})

15-point Gauss-Legendre rule for tetrahedron.
Exact for quartic polynomials (degree 4).

This is a high-accuracy rule suitable for quadratic basis functions.
All weights are positive.
"""
@inline function get_quadrature_points(::Type{Tetrahedron}, ::GaussLegendre{4,V}) where V
    a = 1 / 4
    b1 = (7 + sqrt(15)) / 34
    b2 = (7 - sqrt(15)) / 34
    c1 = (13 - 3 * sqrt(15)) / 34
    c2 = (13 + 3 * sqrt(15)) / 34
    d = (5 - sqrt(15)) / 20
    f = (5 + sqrt(15)) / 20

    w1 = 8 / 405
    w2 = (2665 - 14 * sqrt(15)) / 226800
    w3 = (2665 + 14 * sqrt(15)) / 226800
    w4 = 5 / 567

    return SVector(
        QuadraturePoint(Vec{3}(a, a, a), w1),
        QuadraturePoint(Vec{3}(b1, b1, b1), w2),
        QuadraturePoint(Vec{3}(b1, b1, c1), w2),
        QuadraturePoint(Vec{3}(b1, c1, b1), w2),
        QuadraturePoint(Vec{3}(c1, b1, b1), w2),
        QuadraturePoint(Vec{3}(b2, b2, b2), w3),
        QuadraturePoint(Vec{3}(b2, b2, c2), w3),
        QuadraturePoint(Vec{3}(b2, c2, b2), w3),
        QuadraturePoint(Vec{3}(c2, b2, b2), w3),
        QuadraturePoint(Vec{3}(d, d, f), w4),
        QuadraturePoint(Vec{3}(d, f, d), w4),
        QuadraturePoint(Vec{3}(f, d, d), w4),
        QuadraturePoint(Vec{3}(d, f, f), w4),
        QuadraturePoint(Vec{3}(f, d, f), w4),
        QuadraturePoint(Vec{3}(f, f, d), w4)
    )
end

# ============================================================================
# Legacy symbol-based API (deprecated, kept for backwards compatibility)
# ============================================================================

# These map old :GLTET symbols to new API
get_quadrature_points(::Type{Val{:GLTET1}}) =
    get_quadrature_points(Tetrahedron, GaussLegendre{1}())

get_quadrature_points(::Type{Val{:GLTET4}}) =
    get_quadrature_points(Tetrahedron, GaussLegendre{2}())

get_quadrature_points(::Type{Val{:GLTET5}}) =
    get_quadrature_points(Tetrahedron, GaussLegendre{3}())

get_quadrature_points(::Type{Val{:GLTET15}}) =
    get_quadrature_points(Tetrahedron, GaussLegendre{4}())

# Order queries (deprecated)
get_order(::Type{Val{:GLTET1}}) = 1
get_order(::Type{Val{:GLTET4}}) = 2
get_order(::Type{Val{:GLTET5}}) = 3
get_order(::Type{Val{:GLTET15}}) = 4
