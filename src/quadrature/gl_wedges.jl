# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Gauss-Legendre quadrature rules for wedge (prism) elements.
"""

# ============================================================================
# GaussLegendre{2}: 6-point rules (two variants)
# ============================================================================

"""6-point Gauss-Legendre rule for wedge (default variant)."""
@inline function get_quadrature_points(::Type{Wedge}, ::GaussLegendre{2,:default})
    w = 1/6
    a = sqrt(1/3)

    return SVector(
        QuadraturePoint(Vec{3}((0.5, 0.0, -a)), w),
        QuadraturePoint(Vec{3}((0.0, 0.5, -a)), w),
        QuadraturePoint(Vec{3}((0.5, 0.5, -a)), w),
        QuadraturePoint(Vec{3}((0.5, 0.0, a)), w),
        QuadraturePoint(Vec{3}((0.0, 0.5, a)), w),
        QuadraturePoint(Vec{3}((0.5, 0.5, a)), w)
    )
end

"""6-point Gauss-Legendre rule for wedge (variant B)."""
@inline function get_quadrature_points(::Type{Wedge}, ::GaussLegendre{2,:B})
    w = 1/6
    a = sqrt(1/3)

    return SVector(
        QuadraturePoint(Vec{3}((2/3, 1/6, -a)), w),
        QuadraturePoint(Vec{3}((1/6, 2/3, -a)), w),
        QuadraturePoint(Vec{3}((1/6, 1/6, -a)), w),
        QuadraturePoint(Vec{3}((2/3, 1/6, a)), w),
        QuadraturePoint(Vec{3}((1/6, 2/3, a)), w),
        QuadraturePoint(Vec{3}((1/6, 1/6, a)), w)
    )
end

# ============================================================================
# GaussLegendre{5}: 21-point rule
# ============================================================================

"""21-point Gauss-Legendre rule for wedge."""
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
        QuadraturePoint(Vec{3}((1/3, 1/3, -alpha)), c1 * 9/80),
        QuadraturePoint(Vec{3}((a, a, -alpha)), c1 * (155 + sqrt(15))/2400),
        QuadraturePoint(Vec{3}((1-2a, a, -alpha)), c1 * (155 + sqrt(15))/2400),
        QuadraturePoint(Vec{3}((a, 1-2a, -alpha)), c1 * (155 + sqrt(15))/2400),
        QuadraturePoint(Vec{3}((b, b, -alpha)), c1 * (155 - sqrt(15))/2400),
        QuadraturePoint(Vec{3}((1-2b, b, -alpha)), c1 * (155 - sqrt(15))/2400),
        QuadraturePoint(Vec{3}((b, 1-2b, -alpha)), c1 * (155 - sqrt(15))/2400),

        # z = 0 layer (7 points)
        QuadraturePoint(Vec{3}((1/3, 1/3, 0.0)), c2 * 9/80),
        QuadraturePoint(Vec{3}((a, a, 0.0)), c2 * (155 + sqrt(15))/2400),
        QuadraturePoint(Vec{3}((1-2a, a, 0.0)), c2 * (155 + sqrt(15))/2400),
        QuadraturePoint(Vec{3}((a, 1-2a, 0.0)), c2 * (155 + sqrt(15))/2400),
        QuadraturePoint(Vec{3}((b, b, 0.0)), c2 * (155 - sqrt(15))/2400),
        QuadraturePoint(Vec{3}((1-2b, b, 0.0)), c2 * (155 - sqrt(15))/2400),
        QuadraturePoint(Vec{3}((b, 1-2b, 0.0)), c2 * (155 - sqrt(15))/2400),

        # z = alpha layer (7 points)
        QuadraturePoint(Vec{3}((1/3, 1/3, alpha)), c1 * 9/80),
        QuadraturePoint(Vec{3}((a, a, alpha)), c1 * (155 + sqrt(15))/2400),
        QuadraturePoint(Vec{3}((1-2a, a, alpha)), c1 * (155 + sqrt(15))/2400),
        QuadraturePoint(Vec{3}((a, 1-2a, alpha)), c1 * (155 + sqrt(15))/2400),
        QuadraturePoint(Vec{3}((b, b, alpha)), c1 * (155 - sqrt(15))/2400),
        QuadraturePoint(Vec{3}((1-2b, b, alpha)), c1 * (155 - sqrt(15))/2400),
        QuadraturePoint(Vec{3}((b, 1-2b, alpha)), c1 * (155 - sqrt(15))/2400)
    )
end

