# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using StaticArrays: SVector
using Tensors: Vec

"""
    AbstractQuadratureRule

Abstract base type for all numerical quadrature (integration) schemes.
"""
abstract type AbstractQuadratureRule end

"""
    GaussLegendre{N} <: AbstractQuadratureRule

Gauss-Legendre quadrature with N points per dimension. N points integrate
polynomials of degree 2N-1 exactly.
"""
struct GaussLegendre{N,V} <: AbstractQuadratureRule end

# Constructor with default variant parameter
GaussLegendre{N}() where {N} = GaussLegendre{N,:default}()

"""
    GaussLobatto{N} <: AbstractQuadratureRule

Gauss-Lobatto quadrature with N points per dimension. Includes element boundary
points. N points integrate polynomials of degree 2N-3 exactly.
"""
struct GaussLobatto{N,V} <: AbstractQuadratureRule end

# Constructor with default variant parameter
GaussLobatto{N}() where {N} = GaussLobatto{N,:default}()

"""
    QuadraturePoint{D,T<:Real}

Represents a single quadrature point in D-dimensional parametric space.

# Fields
- `coords::Vec{D,T}`: Location in parametric coordinates
- `weight::T`: Integration weight
"""
struct QuadraturePoint{D,T<:Real}
    coords::Vec{D,T}
    weight::T
end

# Convenience constructor allowing tuple input
QuadraturePoint{D,T}(coords::NTuple{D,T}, weight::T) where {D,T} =
    QuadraturePoint(Vec{D,T}(coords), weight)

# Type aliases for common dimensions
const QuadraturePoint1D{T} = QuadraturePoint{1,T}
const QuadraturePoint2D{T} = QuadraturePoint{2,T}
const QuadraturePoint3D{T} = QuadraturePoint{3,T}

"""
    get_quadrature_points(topology::Type{<:AbstractTopology}, rule::AbstractQuadratureRule)
    -> SVector{N, QuadraturePoint{D,Float64}}

Return the quadrature points and weights for the given topology and rule.
"""
function get_quadrature_points end

"""
    npoints(topology::Type{<:AbstractTopology}, rule::AbstractQuadratureRule) -> Int

Return the number of quadrature points for the given topology and rule.
"""
npoints(topology::Type{<:AbstractTopology}, rule::AbstractQuadratureRule) =
    length(get_quadrature_points(topology, rule))

"""
    default_quadrature(basis_order::Int) -> AbstractQuadratureRule
    default_quadrature(topology::Type{<:AbstractTopology}, basis_order::Int) -> AbstractQuadratureRule
    default_quadrature(topology::Type{<:AbstractTopology}) -> AbstractQuadratureRule

Return the default quadrature rule for the given topology and/or basis order.
Uses `GaussLegendre{basis_order + 1}()` to ensure exact integration.
The single-argument topology form infers the basis order from the node count via
[`_infer_basis_order`](@ref).
"""
function default_quadrature end

# Level 1: Just basis order (simplest, works for most cases).
# Rule of thumb: use order + 1 for stiffness-matrix integration.
default_quadrature(basis_order::Int) = GaussLegendre{basis_order + 1}()

# Level 2: Topology + basis order (handles special cases, can be overridden).
# Default implementation delegates to Level 1.
default_quadrature(::Type{<:AbstractTopology}, basis_order::Int) =
    default_quadrature(basis_order)

# Level 3: Topology type only (infer basis order from node count via the
# `_infer_basis_order` table below). Useful when the basis order is not
# carried explicitly at the call site.

# Helper function to infer basis order from node count.
# This is a heuristic based on standard element types.
function _infer_basis_order end

# 1D Segments
_infer_basis_order(::Type{<:Segment{2}}) = 1   # Linear
_infer_basis_order(::Type{<:Segment{3}}) = 2   # Quadratic

# 2D Triangles (using topology module's parametric types)
_infer_basis_order(::Type{<:Triangle{3}}) = 1   # Linear
_infer_basis_order(::Type{<:Triangle{6}}) = 2   # Quadratic
_infer_basis_order(::Type{<:Triangle{7}}) = 2   # Quadratic with center
_infer_basis_order(::Type{<:Triangle{10}}) = 3  # Cubic

# 2D Quadrilaterals
_infer_basis_order(::Type{<:Quadrilateral{4}}) = 1   # Bilinear
_infer_basis_order(::Type{<:Quadrilateral{8}}) = 2   # Serendipity
_infer_basis_order(::Type{<:Quadrilateral{9}}) = 2   # Biquadratic

# 3D Tetrahedra
_infer_basis_order(::Type{<:Tetrahedron{4}}) = 1    # Linear
_infer_basis_order(::Type{<:Tetrahedron{10}}) = 2   # Quadratic

# 3D Hexahedra
_infer_basis_order(::Type{<:Hexahedron{8}}) = 1     # Trilinear
_infer_basis_order(::Type{<:Hexahedron{20}}) = 2    # Serendipity
_infer_basis_order(::Type{<:Hexahedron{27}}) = 2    # Triquadratic

# 3D Wedges
_infer_basis_order(::Type{<:Wedge{6}}) = 1          # Linear
_infer_basis_order(::Type{<:Wedge{15}}) = 2         # Quadratic

# 3D Pyramids
_infer_basis_order(::Type{<:Pyramid{5}}) = 1        # Linear

# Topology-only dispatch (uses inferred basis order)
default_quadrature(::Type{T}) where {T<:AbstractTopology} =
    default_quadrature(_infer_basis_order(T))

# ============================================================================
# TOPOLOGY TYPE MAPPING - Map parametric topology types to generic quadrature types
# ============================================================================

"""Map parametric topology types to generic quadrature types."""
function _quadrature_topology_type end

# Segments
_quadrature_topology_type(::Type{<:Segment}) = Segment

# Triangles  
_quadrature_topology_type(::Type{<:Triangle}) = Triangle

# Quadrilaterals
_quadrature_topology_type(::Type{<:Quadrilateral}) = Quadrilateral

# Tetrahedra
_quadrature_topology_type(::Type{<:Tetrahedron}) = Tetrahedron

# Hexahedra
_quadrature_topology_type(::Type{<:Hexahedron}) = Hexahedron

# Wedges
_quadrature_topology_type(::Type{<:Wedge}) = Wedge

# Pyramids
_quadrature_topology_type(::Type{<:Pyramid}) = Pyramid
