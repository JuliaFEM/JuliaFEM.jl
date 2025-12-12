# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Gauss-Legendre quadrature rules for tensor-product elements (segments,
quadrilaterals, hexahedra).
"""

# ============================================================================
# 1D Segments
# ============================================================================

"""1-point Gauss-Legendre rule for segment."""
@inline function get_quadrature_points(::Type{Segment}, ::GaussLegendre{1,V}) where V
    return SVector(
        QuadraturePoint(Vec{1}((0.0,)), 2.0)
    )
end

"""2-point Gauss-Legendre rule for segment."""
@inline function get_quadrature_points(::Type{Segment}, ::GaussLegendre{2,V}) where V
    a = 1.0 / sqrt(3.0)
    return SVector(
        QuadraturePoint(Vec{1}((-a,)), 1.0),
        QuadraturePoint(Vec{1}(( a,)), 1.0)
    )
end

"""3-point Gauss-Legendre rule for segment."""
@inline function get_quadrature_points(::Type{Segment}, ::GaussLegendre{3,V}) where V
    a = sqrt(3.0 / 5.0)
    return SVector(
        QuadraturePoint(Vec{1}((-a,)), 5.0/9.0),
        QuadraturePoint(Vec{1}((0.0,)), 8.0/9.0),
        QuadraturePoint(Vec{1}(( a,)), 5.0/9.0)
    )
end

"""4-point Gauss-Legendre rule for segment."""
@inline function get_quadrature_points(::Type{Segment}, ::GaussLegendre{4,V}) where V
    pts, wts = QUAD_DATA[4]
    return SVector(
        QuadraturePoint(Vec{1}((pts[1],)), wts[1]),
        QuadraturePoint(Vec{1}((pts[2],)), wts[2]),
        QuadraturePoint(Vec{1}((pts[3],)), wts[3]),
        QuadraturePoint(Vec{1}((pts[4],)), wts[4])
    )
end

"""5-point Gauss-Legendre rule for segment."""
@inline function get_quadrature_points(::Type{Segment}, ::GaussLegendre{5,V}) where V
    pts, wts = QUAD_DATA[5]
    return SVector(
        QuadraturePoint(Vec{1}((pts[1],)), wts[1]),
        QuadraturePoint(Vec{1}((pts[2],)), wts[2]),
        QuadraturePoint(Vec{1}((pts[3],)), wts[3]),
        QuadraturePoint(Vec{1}((pts[4],)), wts[4]),
        QuadraturePoint(Vec{1}((pts[5],)), wts[5])
    )
end

# ============================================================================
# 2D Quadrilaterals (tensor products)
# ============================================================================

"""1-point Gauss-Legendre rule for quadrilateral."""
@inline function get_quadrature_points(::Type{Quadrilateral}, ::GaussLegendre{1,V}) where V
    return SVector(
        QuadraturePoint(Vec{2}((0.0, 0.0)), 4.0)
    )
end

"""4-point Gauss-Legendre rule for quadrilateral."""
@inline function get_quadrature_points(::Type{Quadrilateral}, ::GaussLegendre{2,V}) where V
    a = 1.0 / sqrt(3.0)
    return SVector(
        QuadraturePoint(Vec{2}((-a, -a)), 1.0),
        QuadraturePoint(Vec{2}(( a, -a)), 1.0),
        QuadraturePoint(Vec{2}((-a, a)), 1.0),
        QuadraturePoint(Vec{2}(( a, a)), 1.0)
    )
end

"""9-point Gauss-Legendre rule for quadrilateral."""
@inline function get_quadrature_points(::Type{Quadrilateral}, ::GaussLegendre{3,V}) where V
    a = sqrt(3.0 / 5.0)
    w1 = 5.0 / 9.0
    w2 = 8.0 / 9.0

    return SVector(
        QuadraturePoint(Vec{2}((-a, -a)), w1*w1),
        QuadraturePoint(Vec{2}((0.0, -a)), w2*w1),
        QuadraturePoint(Vec{2}(( a, -a)), w1*w1),
        QuadraturePoint(Vec{2}((-a, 0.0)), w1*w2),
        QuadraturePoint(Vec{2}((0.0, 0.0)), w2*w2),
        QuadraturePoint(Vec{2}(( a, 0.0)), w1*w2),
        QuadraturePoint(Vec{2}((-a, a)), w1*w1),
        QuadraturePoint(Vec{2}((0.0, a)), w2*w1),
        QuadraturePoint(Vec{2}(( a, a)), w1*w1)
    )
end

"""16-point Gauss-Legendre rule for quadrilateral."""
@inline function get_quadrature_points(::Type{Quadrilateral}, ::GaussLegendre{4,V}) where V
    pts, wts = QUAD_DATA[4]

    result = ntuple(16) do i
        ix = (i - 1) % 4 + 1
        iy = div(i - 1, 4) + 1
        QuadraturePoint(Vec{2}((pts[ix], pts[iy])), wts[ix] * wts[iy])
    end

    return SVector(result)
end

"""25-point Gauss-Legendre rule for quadrilateral."""
@inline function get_quadrature_points(::Type{Quadrilateral}, ::GaussLegendre{5,V}) where V
    pts, wts = QUAD_DATA[5]

    result = ntuple(25) do i
        ix = (i - 1) % 5 + 1
        iy = div(i - 1, 5) + 1
        QuadraturePoint(Vec{2}((pts[ix], pts[iy])), wts[ix] * wts[iy])
    end

    return SVector(result)
end

# ============================================================================
# 3D Hexahedra (tensor products)
# ============================================================================

"""1-point Gauss-Legendre rule for hexahedron."""
@inline function get_quadrature_points(::Type{Hexahedron}, ::GaussLegendre{1,V}) where V
    return SVector(
        QuadraturePoint(Vec{3}((0.0, 0.0, 0.0)), 8.0)
    )
end

"""8-point Gauss-Legendre rule for hexahedron."""
@inline function get_quadrature_points(::Type{Hexahedron}, ::GaussLegendre{2,V}) where V
    a = 1.0 / sqrt(3.0)

    return SVector(
        QuadraturePoint(Vec{3}((-a, -a, -a)), 1.0),
        QuadraturePoint(Vec{3}(( a, -a, -a)), 1.0),
        QuadraturePoint(Vec{3}((-a, a, -a)), 1.0),
        QuadraturePoint(Vec{3}(( a, a, -a)), 1.0),
        QuadraturePoint(Vec{3}((-a, -a, a)), 1.0),
        QuadraturePoint(Vec{3}(( a, -a, a)), 1.0),
        QuadraturePoint(Vec{3}((-a, a, a)), 1.0),
        QuadraturePoint(Vec{3}(( a, a, a)), 1.0)
    )
end

"""27-point Gauss-Legendre rule for hexahedron."""
@inline function get_quadrature_points(::Type{Hexahedron}, ::GaussLegendre{3,V}) where V
    a = sqrt(3.0 / 5.0)
    coords_1d = ((-a, 5.0/9.0), (0.0, 8.0/9.0), (a, 5.0/9.0))

    result = ntuple(27) do i
        ix = (i - 1) % 3 + 1
        iy = div(i - 1, 3) % 3 + 1
        iz = div(i - 1, 9) + 1

        x, wx = coords_1d[ix]
        y, wy = coords_1d[iy]
        z, wz = coords_1d[iz]

        QuadraturePoint(Vec{3}((x, y, z)), wx * wy * wz)
    end

    return SVector(result)
end

"""64-point Gauss-Legendre rule for hexahedron."""
@inline function get_quadrature_points(::Type{Hexahedron}, ::GaussLegendre{4,V}) where V
    pts, wts = QUAD_DATA[4]

    result = ntuple(64) do i
        ix = (i - 1) % 4 + 1
        iy = div(i - 1, 4) % 4 + 1
        iz = div(i - 1, 16) + 1
        QuadraturePoint(Vec{3}((pts[ix], pts[iy], pts[iz])), wts[ix] * wts[iy] * wts[iz])
    end

    return SVector(result)
end

"""125-point Gauss-Legendre rule for hexahedron."""
@inline function get_quadrature_points(::Type{Hexahedron}, ::GaussLegendre{5,V}) where V
    pts, wts = QUAD_DATA[5]

    result = ntuple(125) do i
        ix = (i - 1) % 5 + 1
        iy = div(i - 1, 5) % 5 + 1
        iz = div(i - 1, 25) + 1
        QuadraturePoint(Vec{3}((pts[ix], pts[iy], pts[iz])), wts[ix] * wts[iy] * wts[iz])
    end

    return SVector(result)
end

