# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Gauss-Legendre quadrature rules for wedge (prism) elements.

This file implements quadrature rules for 3D wedge/prism reference elements
formed by extruding a triangle in the z-direction.

# Available Rules
- `GaussLegendre{2}()`: 6-point rule (default variant), exact for linear
- `GaussLegendre{2,:B}()`: 6-point alternative variant, exact for linear
- `GaussLegendre{5}()`: 21-point rule, exact for quintic

# Notes
- Wedge is tensor product of triangle (ξ,η) and segment (ζ)
- Rules are typically products of triangle rules × 1D Gauss rules
- 6-point rules: 3 triangle points × 2 segment points
- 21-point rule: 7 triangle points × 3 segment points

See also: [`GaussLegendre`](@ref), [`QuadraturePoint`](@ref)
"""

# ============================================================================
# GaussLegendre{2}: 6-point rules (two variants)
# ============================================================================

"""
    get_quadrature_points(::Type{Wedge}, ::GaussLegendre{2})

6-point Gauss-Legendre rule for wedge (default variant).
Exact for linear polynomials (degree 1).

Tensor product: 3 triangle points × 2 segment points.
"""
@inline function get_quadrature_points(::Type{Wedge}, ::GaussLegendre{2,:default})
    w = 1/6
    a = sqrt(1/3)

    return SVector(
        QuadraturePoint(Vec{3}(0.5, 0.0, -a), w),
        QuadraturePoint(Vec{3}(0.0, 0.5, -a), w),
        QuadraturePoint(Vec{3}(0.5, 0.5, -a), w),
        QuadraturePoint(Vec{3}(0.5, 0.0,  a), w),
        QuadraturePoint(Vec{3}(0.0, 0.5,  a), w),
        QuadraturePoint(Vec{3}(0.5, 0.5,  a), w)
    )
end

"""
    get_quadrature_points(::Type{Wedge}, ::GaussLegendre{2,:B})

6-point Gauss-Legendre rule for wedge (variant B).
Exact for linear polynomials (degree 1).

Alternative triangle point distribution.
"""
@inline function get_quadrature_points(::Type{Wedge}, ::GaussLegendre{2,:B})
    w = 1/6
    a = sqrt(1/3)

    return SVector(
        QuadraturePoint(Vec{3}(2/3, 1/6, -a), w),
        QuadraturePoint(Vec{3}(1/6, 2/3, -a), w),
        QuadraturePoint(Vec{3}(1/6, 1/6, -a), w),
        QuadraturePoint(Vec{3}(2/3, 1/6,  a), w),
        QuadraturePoint(Vec{3}(1/6, 2/3,  a), w),
        QuadraturePoint(Vec{3}(1/6, 1/6,  a), w)
    )
end

# ============================================================================
# GaussLegendre{5}: 21-point rule
# ============================================================================

"""
    get_quadrature_points(::Type{Wedge}, ::GaussLegendre{5})

21-point Gauss-Legendre rule for wedge.
Exact for quintic polynomials (degree 5).

Tensor product: 7 triangle points × 3 segment points.
High-accuracy rule suitable for quadratic basis functions.
"""
@inline function get_quadrature_points(::Type{Wedge}, ::GaussLegendre{5,V}) where V
    # 1D Gauss-Legendre 3-point rule
    alpha = sqrt(3/5)
    c1 = 5/9
    c2 = 8/9

    # Triangle 7-point rule parameters
    a = (6 + sqrt(15)) / 21
    b = (6 - sqrt(15)) / 21

    return SVector(
        # z = -alpha layer (7 points)
        QuadraturePoint(Vec{3}(1/3, 1/3, -alpha), c1 * 9/80),
        QuadraturePoint(Vec{3}(a, a, -alpha), c1 * (155 + sqrt(15))/2400),
        QuadraturePoint(Vec{3}(1-2a, a, -alpha), c1 * (155 + sqrt(15))/2400),
        QuadraturePoint(Vec{3}(a, 1-2a, -alpha), c1 * (155 + sqrt(15))/2400),
        QuadraturePoint(Vec{3}(b, b, -alpha), c1 * (155 - sqrt(15))/2400),
        QuadraturePoint(Vec{3}(1-2b, b, -alpha), c1 * (155 - sqrt(15))/2400),
        QuadraturePoint(Vec{3}(b, 1-2b, -alpha), c1 * (155 - sqrt(15))/2400),

        # z = 0 layer (7 points)
        QuadraturePoint(Vec{3}(1/3, 1/3, 0.0), c2 * 9/80),
        QuadraturePoint(Vec{3}(a, a, 0.0), c2 * (155 + sqrt(15))/2400),
        QuadraturePoint(Vec{3}(1-2a, a, 0.0), c2 * (155 + sqrt(15))/2400),
        QuadraturePoint(Vec{3}(a, 1-2a, 0.0), c2 * (155 + sqrt(15))/2400),
        QuadraturePoint(Vec{3}(b, b, 0.0), c2 * (155 - sqrt(15))/2400),
        QuadraturePoint(Vec{3}(1-2b, b, 0.0), c2 * (155 - sqrt(15))/2400),
        QuadraturePoint(Vec{3}(b, 1-2b, 0.0), c2 * (155 - sqrt(15))/2400),

        # z = alpha layer (7 points)
        QuadraturePoint(Vec{3}(1/3, 1/3, alpha), c1 * 9/80),
        QuadraturePoint(Vec{3}(a, a, alpha), c1 * (155 + sqrt(15))/2400),
        QuadraturePoint(Vec{3}(1-2a, a, alpha), c1 * (155 + sqrt(15))/2400),
        QuadraturePoint(Vec{3}(a, 1-2a, alpha), c1 * (155 + sqrt(15))/2400),
        QuadraturePoint(Vec{3}(b, b, alpha), c1 * (155 - sqrt(15))/2400),
        QuadraturePoint(Vec{3}(1-2b, b, alpha), c1 * (155 - sqrt(15))/2400),
        QuadraturePoint(Vec{3}(b, 1-2b, alpha), c1 * (155 - sqrt(15))/2400)
    )
end

# ============================================================================
# Legacy symbol-based API (deprecated, kept for backwards compatibility)
# ============================================================================

# These map old :GLWED symbols to new API
get_quadrature_points(::Type{Val{:GLWED6}}) =
    get_quadrature_points(Wedge, GaussLegendre{2}())

get_quadrature_points(::Type{Val{:GLWED6B}}) =
    get_quadrature_points(Wedge, GaussLegendre{2,:B}())

get_quadrature_points(::Type{Val{:GLWED21}}) =
    get_quadrature_points(Wedge, GaussLegendre{5}())

# Order queries (deprecated)
get_order(::Type{Val{:GLWED6}}) = 1
get_order(::Type{Val{:GLWED6B}}) = 1
get_order(::Type{Val{:GLWED21}}) = 5
