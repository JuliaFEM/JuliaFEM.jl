# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Gauss-Legendre quadrature rules for tensor-product elements.

This file implements quadrature rules for elements formed by tensor products
of 1D Gauss-Legendre rules:
- **Segments** (1D): Direct 1D Gauss rules
- **Quadrilaterals** (2D): Product of two 1D rules
- **Hexahedra** (3D): Product of three 1D rules

# Available Rules

## Segments (1D)
- `GaussLegendre{1}()`: 1-point, exact for linear
- `GaussLegendre{2}()`: 2-point, exact for cubic
- `GaussLegendre{3}()`: 3-point, exact for quintic
- `GaussLegendre{4}()`: 4-point, exact for degree 7
- `GaussLegendre{5}()`: 5-point, exact for degree 9

## Quadrilaterals (2D)
- `GaussLegendre{1}()`: 1×1 = 1 point
- `GaussLegendre{2}()`: 2×2 = 4 points (standard for bilinear)
- `GaussLegendre{3}()`: 3×3 = 9 points (standard for biquadratic)
- `GaussLegendre{4}()`: 4×4 = 16 points
- `GaussLegendre{5}()`: 5×5 = 25 points

## Hexahedra (3D)
- `GaussLegendre{1}()`: 1×1×1 = 1 point
- `GaussLegendre{2}()`: 2×2×2 = 8 points (standard for trilinear)
- `GaussLegendre{3}()`: 3×3×3 = 27 points (standard for triquadratic)
- `GaussLegendre{4}()`: 4×4×4 = 64 points
- `GaussLegendre{5}()`: 5×5×5 = 125 points

# Notes
- Tensor products are formed from 1D Gauss-Legendre points in quaddata.jl
- N points per dimension → N^D total points
- Reference domains: Segment [-1,1], Quad [-1,1]², Hex [-1,1]³
- All implementations use @inline for zero-allocation

See also: [`GaussLegendre`](@ref), [`QuadraturePoint`](@ref)
"""

# ============================================================================
# 1D Segments
# ============================================================================

"""
    get_quadrature_points(::Type{Segment}, ::GaussLegendre{1})

1-point Gauss-Legendre rule for segment.
Exact for linear polynomials (degree 1).
"""
@inline function get_quadrature_points(::Type{Segment}, ::GaussLegendre{1,V}) where V
    return SVector(
        QuadraturePoint(Vec{1}(0.0), 2.0)
    )
end

"""
    get_quadrature_points(::Type{Segment}, ::GaussLegendre{2})

2-point Gauss-Legendre rule for segment.
Exact for cubic polynomials (degree 3).
"""
@inline function get_quadrature_points(::Type{Segment}, ::GaussLegendre{2,V}) where V
    a = 1.0 / sqrt(3.0)
    return SVector(
        QuadraturePoint(Vec{1}(-a), 1.0),
        QuadraturePoint(Vec{1}( a), 1.0)
    )
end

"""
    get_quadrature_points(::Type{Segment}, ::GaussLegendre{3})

3-point Gauss-Legendre rule for segment.
Exact for quintic polynomials (degree 5).
"""
@inline function get_quadrature_points(::Type{Segment}, ::GaussLegendre{3,V}) where V
    a = sqrt(3.0 / 5.0)
    return SVector(
        QuadraturePoint(Vec{1}(-a), 5.0/9.0),
        QuadraturePoint(Vec{1}(0.0), 8.0/9.0),
        QuadraturePoint(Vec{1}( a), 5.0/9.0)
    )
end

"""
    get_quadrature_points(::Type{Segment}, ::GaussLegendre{4})

4-point Gauss-Legendre rule for segment.
Exact for degree 7 polynomials.
"""
@inline function get_quadrature_points(::Type{Segment}, ::GaussLegendre{4,V}) where V
    pts, wts = QUAD_DATA[4]
    return SVector(
        QuadraturePoint(Vec{1}(pts[1]), wts[1]),
        QuadraturePoint(Vec{1}(pts[2]), wts[2]),
        QuadraturePoint(Vec{1}(pts[3]), wts[3]),
        QuadraturePoint(Vec{1}(pts[4]), wts[4])
    )
end

"""
    get_quadrature_points(::Type{Segment}, ::GaussLegendre{5})

5-point Gauss-Legendre rule for segment.
Exact for degree 9 polynomials.
"""
@inline function get_quadrature_points(::Type{Segment}, ::GaussLegendre{5,V}) where V
    pts, wts = QUAD_DATA[5]
    return SVector(
        QuadraturePoint(Vec{1}(pts[1]), wts[1]),
        QuadraturePoint(Vec{1}(pts[2]), wts[2]),
        QuadraturePoint(Vec{1}(pts[3]), wts[3]),
        QuadraturePoint(Vec{1}(pts[4]), wts[4]),
        QuadraturePoint(Vec{1}(pts[5]), wts[5])
    )
end

# ============================================================================
# 2D Quadrilaterals (tensor products)
# ============================================================================

"""
    get_quadrature_points(::Type{Quadrilateral}, ::GaussLegendre{1})

1×1 = 1-point Gauss-Legendre rule for quadrilateral.
"""
@inline function get_quadrature_points(::Type{Quadrilateral}, ::GaussLegendre{1,V}) where V
    return SVector(
        QuadraturePoint(Vec{2}(0.0, 0.0), 4.0)
    )
end

"""
    get_quadrature_points(::Type{Quadrilateral}, ::GaussLegendre{2})

2×2 = 4-point Gauss-Legendre rule for quadrilateral.
Standard rule for bilinear elements (Quad4).
"""
@inline function get_quadrature_points(::Type{Quadrilateral}, ::GaussLegendre{2,V}) where V
    a = 1.0 / sqrt(3.0)
    return SVector(
        QuadraturePoint(Vec{2}(-a, -a), 1.0),
        QuadraturePoint(Vec{2}( a, -a), 1.0),
        QuadraturePoint(Vec{2}(-a,  a), 1.0),
        QuadraturePoint(Vec{2}( a,  a), 1.0)
    )
end

"""
    get_quadrature_points(::Type{Quadrilateral}, ::GaussLegendre{3})

3×3 = 9-point Gauss-Legendre rule for quadrilateral.
Standard rule for biquadratic elements (Quad9).
"""
@inline function get_quadrature_points(::Type{Quadrilateral}, ::GaussLegendre{3,V}) where V
    a = sqrt(3.0 / 5.0)
    w1 = 5.0 / 9.0
    w2 = 8.0 / 9.0

    return SVector(
        QuadraturePoint(Vec{2}(-a, -a), w1*w1),
        QuadraturePoint(Vec{2}(0.0, -a), w2*w1),
        QuadraturePoint(Vec{2}( a, -a), w1*w1),
        QuadraturePoint(Vec{2}(-a, 0.0), w1*w2),
        QuadraturePoint(Vec{2}(0.0, 0.0), w2*w2),
        QuadraturePoint(Vec{2}( a, 0.0), w1*w2),
        QuadraturePoint(Vec{2}(-a,  a), w1*w1),
        QuadraturePoint(Vec{2}(0.0,  a), w2*w1),
        QuadraturePoint(Vec{2}( a,  a), w1*w1)
    )
end

"""
    get_quadrature_points(::Type{Quadrilateral}, ::GaussLegendre{4})

4×4 = 16-point Gauss-Legendre rule for quadrilateral.
"""
@inline function get_quadrature_points(::Type{Quadrilateral}, ::GaussLegendre{4,V}) where V
    pts, wts = QUAD_DATA[4]

    result = ntuple(16) do i
        ix = (i - 1) % 4 + 1
        iy = div(i - 1, 4) + 1
        QuadraturePoint(Vec{2}(pts[ix], pts[iy]), wts[ix] * wts[iy])
    end

    return SVector(result)
end

"""
    get_quadrature_points(::Type{Quadrilateral}, ::GaussLegendre{5})

5×5 = 25-point Gauss-Legendre rule for quadrilateral.
"""
@inline function get_quadrature_points(::Type{Quadrilateral}, ::GaussLegendre{5,V}) where V
    pts, wts = QUAD_DATA[5]

    result = ntuple(25) do i
        ix = (i - 1) % 5 + 1
        iy = div(i - 1, 5) + 1
        QuadraturePoint(Vec{2}(pts[ix], pts[iy]), wts[ix] * wts[iy])
    end

    return SVector(result)
end

# ============================================================================
# 3D Hexahedra (tensor products)
# ============================================================================

"""
    get_quadrature_points(::Type{Hexahedron}, ::GaussLegendre{1})

1×1×1 = 1-point Gauss-Legendre rule for hexahedron.
"""
@inline function get_quadrature_points(::Type{Hexahedron}, ::GaussLegendre{1,V}) where V
    return SVector(
        QuadraturePoint(Vec{3}(0.0, 0.0, 0.0), 8.0)
    )
end

"""
    get_quadrature_points(::Type{Hexahedron}, ::GaussLegendre{2})

2×2×2 = 8-point Gauss-Legendre rule for hexahedron.
Standard rule for trilinear elements (Hex8).
"""
@inline function get_quadrature_points(::Type{Hexahedron}, ::GaussLegendre{2,V}) where V
    a = 1.0 / sqrt(3.0)

    return SVector(
        QuadraturePoint(Vec{3}(-a, -a, -a), 1.0),
        QuadraturePoint(Vec{3}( a, -a, -a), 1.0),
        QuadraturePoint(Vec{3}(-a,  a, -a), 1.0),
        QuadraturePoint(Vec{3}( a,  a, -a), 1.0),
        QuadraturePoint(Vec{3}(-a, -a,  a), 1.0),
        QuadraturePoint(Vec{3}( a, -a,  a), 1.0),
        QuadraturePoint(Vec{3}(-a,  a,  a), 1.0),
        QuadraturePoint(Vec{3}( a,  a,  a), 1.0)
    )
end

"""
    get_quadrature_points(::Type{Hexahedron}, ::GaussLegendre{3})

3×3×3 = 27-point Gauss-Legendre rule for hexahedron.
Standard rule for triquadratic elements (Hex27).
"""
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

        QuadraturePoint(Vec{3}(x, y, z), wx * wy * wz)
    end

    return SVector(result)
end

"""
    get_quadrature_points(::Type{Hexahedron}, ::GaussLegendre{4})

4×4×4 = 64-point Gauss-Legendre rule for hexahedron.
"""
@inline function get_quadrature_points(::Type{Hexahedron}, ::GaussLegendre{4,V}) where V
    pts, wts = QUAD_DATA[4]

    result = ntuple(64) do i
        ix = (i - 1) % 4 + 1
        iy = div(i - 1, 4) % 4 + 1
        iz = div(i - 1, 16) + 1
        QuadraturePoint(Vec{3}(pts[ix], pts[iy], pts[iz]), wts[ix] * wts[iy] * wts[iz])
    end

    return SVector(result)
end

"""
    get_quadrature_points(::Type{Hexahedron}, ::GaussLegendre{5})

5×5×5 = 125-point Gauss-Legendre rule for hexahedron.
"""
@inline function get_quadrature_points(::Type{Hexahedron}, ::GaussLegendre{5,V}) where V
    pts, wts = QUAD_DATA[5]

    result = ntuple(125) do i
        ix = (i - 1) % 5 + 1
        iy = div(i - 1, 5) % 5 + 1
        iz = div(i - 1, 25) + 1
        QuadraturePoint(Vec{3}(pts[ix], pts[iy], pts[iz]), wts[ix] * wts[iy] * wts[iz])
    end

    return SVector(result)
end

# ============================================================================
# Legacy symbol-based API (deprecated, kept for backwards compatibility)
# ============================================================================

# Helper function for old API (generates zip of tuples)
function _legacy_tensor_product(w::Tuple, p::Tuple, dim::Int)
    @assert length(w) == length(p)
    N = length(w)
    weights = Float64[]
    points = NTuple{dim, Float64}[]
    for i in CartesianIndices(ntuple(i -> 1:N, dim))
        push!(weights, prod(w[k] for k in Tuple(i)))
        push!(points, ntuple(k -> p[i[k]], dim))
    end
    return zip(Tuple(weights), Tuple(points))
end

# Generate legacy symbol-based rules programmatically
const _LEGACY_NAMES = [:GLSEG, :GLQUAD, :GLHEX]
const _LEGACY_DIMS = (1, 2, 3)

for n in 1:length(QUAD_DATA)
    points, weights = QUAD_DATA[n]
    order = 2(n-1) + 1

    for (dim, name) in zip(_LEGACY_DIMS, _LEGACY_NAMES)
        n_points = length(points)^dim
        quadname = QuoteNode(Symbol(string(name, n_points)))
        z = _legacy_tensor_product(weights, points, dim)

        @eval begin
            get_quadrature_points(::Type{Val{$(quadname)}}) = $z
            get_order(::Type{Val{$(quadname)}}) = $order
        end
    end
end
