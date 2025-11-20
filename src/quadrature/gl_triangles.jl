# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Gauss-Legendre quadrature rules for triangular elements.

This file implements quadrature rules for 2D triangular reference elements
in the parametric domain [0,1]² with constraint ξ + η ≤ 1.

# Available Rules
- `GaussLegendre{1}()`: 1-point rule (centroid), exact for linear
- `GaussLegendre{2}()`: 3-point rule, exact for quadratic (default variant)
- `GaussLegendre{2,:B}()`: 3-point alternative (midpoint variant)
- `GaussLegendre{3}()`: 4-point rule, exact for cubic (default variant)
- `GaussLegendre{3,:B}()`: 4-point alternative (has negative weight!)
- `GaussLegendre{4}()`: 6-point rule, exact for quartic
- `GaussLegendre{5}()`: 7-point rule, exact for quintic
- `GaussLegendre{6}()`: 12-point rule, exact for 6th degree

# Notes
- Reference triangle has vertices at (0,0), (1,0), (0,1)
- All weights sum to 0.5 (area of reference triangle)
- Variant :B rules exist for historical reasons, default is recommended
- Negative weights in GLTRI4B may cause issues, use default GaussLegendre{3}()

See also: [`GaussLegendre`](@ref), [`QuadraturePoint`](@ref)
"""

# ============================================================================
# GaussLegendre{1}: 1-point rule (centroid)
# ============================================================================

"""
    get_quadrature_points(::Type{Triangle}, ::GaussLegendre{1})

1-point Gauss-Legendre rule for triangle (centroid).
Exact for linear polynomials (degree 1).
"""
@inline function get_quadrature_points(::Type{Triangle}, ::GaussLegendre{1,V}) where V
    return SVector(
        QuadraturePoint(Vec{2}(1/3, 1/3), 0.5)
    )
end

# ============================================================================
# GaussLegendre{2}: 3-point rules (two variants)
# ============================================================================

"""
    get_quadrature_points(::Type{Triangle}, ::GaussLegendre{2})

3-point Gauss-Legendre rule for triangle (default variant).
Exact for quadratic polynomials (degree 2).
Points located at edge centers of sub-triangles.
"""
@inline function get_quadrature_points(::Type{Triangle}, ::GaussLegendre{2,:default})
    w = 1/6
    return SVector(
        QuadraturePoint(Vec{2}(2/3, 1/6), w),
        QuadraturePoint(Vec{2}(1/6, 2/3), w),
        QuadraturePoint(Vec{2}(1/6, 1/6), w)
    )
end

"""
    get_quadrature_points(::Type{Triangle}, ::GaussLegendre{2,:B})

3-point Gauss-Legendre rule for triangle (variant B).
Exact for quadratic polynomials (degree 2).
Points located at edge midpoints.
"""
@inline function get_quadrature_points(::Type{Triangle}, ::GaussLegendre{2,:B})
    w = 1/6
    return SVector(
        QuadraturePoint(Vec{2}(0.0, 1/2), w),
        QuadraturePoint(Vec{2}(1/2, 0.0), w),
        QuadraturePoint(Vec{2}(1/2, 1/2), w)
    )
end

# ============================================================================
# GaussLegendre{3}: 4-point rules (two variants)
# ============================================================================

"""
    get_quadrature_points(::Type{Triangle}, ::GaussLegendre{3})

4-point Gauss-Legendre rule for triangle (default variant).
Exact for cubic polynomials (degree 3).
All weights positive.
"""
@inline function get_quadrature_points(::Type{Triangle}, ::GaussLegendre{3,:default})
    return SVector(
        QuadraturePoint(Vec{2}(0.15505102572168219, 0.17855872826361642), 0.15902069087198858),
        QuadraturePoint(Vec{2}(0.64494897427831781, 0.07503111022260812), 0.09097930912801142),
        QuadraturePoint(Vec{2}(0.15505102572168219, 0.66639024601470139), 0.15902069087198858),
        QuadraturePoint(Vec{2}(0.64494897427831781, 0.28001991549907407), 0.09097930912801142)
    )
end

"""
    get_quadrature_points(::Type{Triangle}, ::GaussLegendre{3,:B})

4-point Gauss-Legendre rule for triangle (variant B).
Exact for cubic polynomials (degree 3).

**WARNING:** This rule has one negative weight (-27/96), which may cause
numerical issues. Use default variant instead unless specifically needed.
"""
@inline function get_quadrature_points(::Type{Triangle}, ::GaussLegendre{3,:B})
    return SVector(
        QuadraturePoint(Vec{2}(1/3, 1/3), -27/96),  # Negative weight!
        QuadraturePoint(Vec{2}(1/5, 1/5),  25/96),
        QuadraturePoint(Vec{2}(1/5, 3/5),  25/96),
        QuadraturePoint(Vec{2}(3/5, 1/5),  25/96)
    )
end

# ============================================================================
# GaussLegendre{4}: 6-point rule
# ============================================================================

"""
    get_quadrature_points(::Type{Triangle}, ::GaussLegendre{4})

6-point Gauss-Legendre rule for triangle.
Exact for quartic polynomials (degree 4).
"""
@inline function get_quadrature_points(::Type{Triangle}, ::GaussLegendre{4,V}) where V
    P1 = 0.11169079483905
    P2 = 0.0549758718227661
    A = 0.445948490915965
    B = 0.091576213509771

    return SVector(
        QuadraturePoint(Vec{2}(B, B), P2),
        QuadraturePoint(Vec{2}(1.0 - 2.0*B, B), P2),
        QuadraturePoint(Vec{2}(B, 1.0 - 2.0*B), P2),
        QuadraturePoint(Vec{2}(A, 1.0 - 2.0*A), P1),
        QuadraturePoint(Vec{2}(A, A), P1),
        QuadraturePoint(Vec{2}(1.0 - 2.0*A, A), P1)
    )
end

# ============================================================================
# GaussLegendre{5}: 7-point rule
# ============================================================================

"""
    get_quadrature_points(::Type{Triangle}, ::GaussLegendre{5})

7-point Gauss-Legendre rule for triangle.
Exact for quintic polynomials (degree 5).
Includes centroid point.
"""
@inline function get_quadrature_points(::Type{Triangle}, ::GaussLegendre{5,V}) where V
    A = 0.470142064105115
    B = 0.101286507323456
    P1 = 0.066197076394253
    P2 = 0.062969590272413

    return SVector(
        QuadraturePoint(Vec{2}(1/3, 1/3), 9/80),
        QuadraturePoint(Vec{2}(A, A), P1),
        QuadraturePoint(Vec{2}(1.0 - 2.0*A, A), P1),
        QuadraturePoint(Vec{2}(A, 1.0 - 2.0*A), P1),
        QuadraturePoint(Vec{2}(B, B), P2),
        QuadraturePoint(Vec{2}(1.0 - 2.0*B, B), P2),
        QuadraturePoint(Vec{2}(B, 1.0 - 2.0*B), P2)
    )
end

# ============================================================================
# GaussLegendre{6}: 12-point rule
# ============================================================================

"""
    get_quadrature_points(::Type{Triangle}, ::GaussLegendre{6})

12-point Gauss-Legendre rule for triangle.
Exact for 6th degree polynomials.
"""
@inline function get_quadrature_points(::Type{Triangle}, ::GaussLegendre{6,V}) where V
    A = 0.063089014491502
    B = 0.249286745170910
    C = 0.310352451033785
    D = 0.053145049844816
    P1 = 0.025422453185103
    P2 = 0.058393137863189
    P3 = 0.041425537809187

    return SVector(
        QuadraturePoint(Vec{2}(A, A), P1),
        QuadraturePoint(Vec{2}(1.0 - 2.0*A, A), P1),
        QuadraturePoint(Vec{2}(A, 1.0 - 2.0*A), P1),
        QuadraturePoint(Vec{2}(B, B), P2),
        QuadraturePoint(Vec{2}(1.0 - 2.0*B, B), P2),
        QuadraturePoint(Vec{2}(B, 1.0 - 2.0*B), P2),
        QuadraturePoint(Vec{2}(C, D), P3),
        QuadraturePoint(Vec{2}(D, C), P3),
        QuadraturePoint(Vec{2}(1.0 - C - D, C), P3),
        QuadraturePoint(Vec{2}(1.0 - C - D, D), P3),
        QuadraturePoint(Vec{2}(C, 1.0 - C - D), P3),
        QuadraturePoint(Vec{2}(D, 1.0 - C - D), P3)
    )
end

# ============================================================================
# Legacy symbol-based API (deprecated, kept for backwards compatibility)
# ============================================================================

# These map old :GLTRI symbols to new API
get_quadrature_points(::Type{Val{:GLTRI1}}) =
    get_quadrature_points(Triangle, GaussLegendre{1}())

get_quadrature_points(::Type{Val{:GLTRI3}}) =
    get_quadrature_points(Triangle, GaussLegendre{2}())

get_quadrature_points(::Type{Val{:GLTRI3B}}) =
    get_quadrature_points(Triangle, GaussLegendre{2,:B}())

get_quadrature_points(::Type{Val{:GLTRI4}}) =
    get_quadrature_points(Triangle, GaussLegendre{3}())

get_quadrature_points(::Type{Val{:GLTRI4B}}) =
    get_quadrature_points(Triangle, GaussLegendre{3,:B}())

get_quadrature_points(::Type{Val{:GLTRI6}}) =
    get_quadrature_points(Triangle, GaussLegendre{4}())

get_quadrature_points(::Type{Val{:GLTRI7}}) =
    get_quadrature_points(Triangle, GaussLegendre{5}())

get_quadrature_points(::Type{Val{:GLTRI12}}) =
    get_quadrature_points(Triangle, GaussLegendre{6}())

# Order queries (deprecated)
get_order(::Type{Val{:GLTRI1}}) = 1
get_order(::Type{Val{:GLTRI3}}) = 2
get_order(::Type{Val{:GLTRI3B}}) = 2
get_order(::Type{Val{:GLTRI4}}) = 3
get_order(::Type{Val{:GLTRI4B}}) = 3
get_order(::Type{Val{:GLTRI6}}) = 4
get_order(::Type{Val{:GLTRI7}}) = 5
get_order(::Type{Val{:GLTRI12}}) = 6
