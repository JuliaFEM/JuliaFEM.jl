# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Gauss-Legendre quadrature rules for triangular elements.
"""

# ============================================================================
# GaussLegendre{1}: 1-point rule (centroid)
# ============================================================================

"""1-point Gauss-Legendre rule for triangle."""
@inline function get_quadrature_points(::Type{Triangle}, ::GaussLegendre{1,V}) where V
    return SVector(
        QuadraturePoint(Vec{2}((1/3, 1/3)), 0.5)
    )
end

# ============================================================================
# GaussLegendre{2}: 3-point rules (two variants)
# ============================================================================

"""3-point Gauss-Legendre rule for triangle (default variant)."""
@inline function get_quadrature_points(::Type{Triangle}, ::GaussLegendre{2,:default})
    w = 1/6
    return SVector(
        QuadraturePoint(Vec{2}((2/3, 1/6)), w),
        QuadraturePoint(Vec{2}((1/6, 2/3)), w),
        QuadraturePoint(Vec{2}((1/6, 1/6)), w)
    )
end

"""3-point Gauss-Legendre rule for triangle (variant B)."""
@inline function get_quadrature_points(::Type{Triangle}, ::GaussLegendre{2,:B})
    w = 1/6
    return SVector(
        QuadraturePoint(Vec{2}((0.0, 1/2)), w),
        QuadraturePoint(Vec{2}((1/2, 0.0)), w),
        QuadraturePoint(Vec{2}((1/2, 1/2)), w)
    )
end

# ============================================================================
# GaussLegendre{3}: 4-point rules (two variants)
# ============================================================================

"""4-point Gauss-Legendre rule for triangle (default variant)."""
@inline function get_quadrature_points(::Type{Triangle}, ::GaussLegendre{3,:default})
    return SVector(
        QuadraturePoint(Vec{2}((0.15505102572168219, 0.17855872826361642)), 0.15902069087198858),
        QuadraturePoint(Vec{2}((0.64494897427831781, 0.07503111022260812)), 0.09097930912801142),
        QuadraturePoint(Vec{2}((0.15505102572168219, 0.66639024601470139)), 0.15902069087198858),
        QuadraturePoint(Vec{2}((0.64494897427831781, 0.28001991549907407)), 0.09097930912801142)
    )
end

"""4-point Gauss-Legendre rule for triangle (variant B). Has negative weight."""
@inline function get_quadrature_points(::Type{Triangle}, ::GaussLegendre{3,:B})
    return SVector(
        QuadraturePoint(Vec{2}((1/3, 1/3)), -27/96),  # Negative weight!
        QuadraturePoint(Vec{2}((1/5, 1/5)),  25/96),
        QuadraturePoint(Vec{2}((1/5, 3/5)),  25/96),
        QuadraturePoint(Vec{2}((3/5, 1/5)),  25/96)
    )
end

# ============================================================================
# GaussLegendre{4}: 6-point rule
# ============================================================================

"""6-point Gauss-Legendre rule for triangle."""
@inline function get_quadrature_points(::Type{Triangle}, ::GaussLegendre{4,V}) where V
    P1 = 0.11169079483905
    P2 = 0.0549758718227661
    A = 0.445948490915965
    B = 0.091576213509771

    return SVector(
        QuadraturePoint(Vec{2}((B, B)), P2),
        QuadraturePoint(Vec{2}((1.0 - 2.0*B, B)), P2),
        QuadraturePoint(Vec{2}((B, 1.0 - 2.0*B)), P2),
        QuadraturePoint(Vec{2}((A, 1.0 - 2.0*A)), P1),
        QuadraturePoint(Vec{2}((A, A)), P1),
        QuadraturePoint(Vec{2}((1.0 - 2.0*A, A)), P1)
    )
end

# ============================================================================
# GaussLegendre{5}: 7-point rule
# ============================================================================

"""7-point Gauss-Legendre rule for triangle."""
@inline function get_quadrature_points(::Type{Triangle}, ::GaussLegendre{5,V}) where V
    A = 0.470142064105115
    B = 0.101286507323456
    P1 = 0.066197076394253
    P2 = 0.062969590272413

    return SVector(
        QuadraturePoint(Vec{2}((1/3, 1/3)), 9/80),
        QuadraturePoint(Vec{2}((A, A)), P1),
        QuadraturePoint(Vec{2}((1.0 - 2.0*A, A)), P1),
        QuadraturePoint(Vec{2}((A, 1.0 - 2.0*A)), P1),
        QuadraturePoint(Vec{2}((B, B)), P2),
        QuadraturePoint(Vec{2}((1.0 - 2.0*B, B)), P2),
        QuadraturePoint(Vec{2}((B, 1.0 - 2.0*B)), P2)
    )
end

# ============================================================================
# GaussLegendre{6}: 12-point rule
# ============================================================================

"""12-point Gauss-Legendre rule for triangle."""
@inline function get_quadrature_points(::Type{Triangle}, ::GaussLegendre{6,V}) where V
    A = 0.063089014491502
    B = 0.249286745170910
    C = 0.310352451033785
    D = 0.053145049844816
    P1 = 0.025422453185103
    P2 = 0.058393137863189
    P3 = 0.041425537809187

    return SVector(
        QuadraturePoint(Vec{2}((A, A)), P1),
        QuadraturePoint(Vec{2}((1.0 - 2.0*A, A)), P1),
        QuadraturePoint(Vec{2}((A, 1.0 - 2.0*A)), P1),
        QuadraturePoint(Vec{2}((B, B)), P2),
        QuadraturePoint(Vec{2}((1.0 - 2.0*B, B)), P2),
        QuadraturePoint(Vec{2}((B, 1.0 - 2.0*B)), P2),
        QuadraturePoint(Vec{2}((C, D)), P3),
        QuadraturePoint(Vec{2}((D, C)), P3),
        QuadraturePoint(Vec{2}((1.0 - C - D, C)), P3),
        QuadraturePoint(Vec{2}((1.0 - C - D, D)), P3),
        QuadraturePoint(Vec{2}((C, 1.0 - C - D)), P3),
        QuadraturePoint(Vec{2}((D, 1.0 - C - D)), P3)
    )
end

