# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using StaticArrays: SVector
using Tensors: Vec

"""
    AbstractQuadratureRule

Abstract base type for all numerical quadrature (integration) schemes.

A quadrature rule defines how to numerically integrate over a reference element
by specifying quadrature point locations and weights. Quadrature rules are
independent of element topology (though the number of points may depend on
polynomial order and element geometry).

# Concrete Subtypes
- [`GaussLegendre`](@ref) - Gauss-Legendre quadrature (open rules)
- [`GaussLobatto`](@ref) - Gauss-Lobatto quadrature (includes element boundaries)

# Key Properties
- Quadrature points (locations in parametric space)
- Weights
- Accuracy order (exact integration up to polynomial degree)

# Examples

    # Gauss-Legendre quadrature
    rule = GaussLegendre{2}()  # 2 points per dimension

    # Gauss-Lobatto quadrature (includes endpoints)
    rule = GaussLobatto{3}()   # 3 points per dimension

    # Get quadrature points for a topology
    points = get_quadrature_points(Triangle, rule)

See also: [`GaussLegendre`](@ref), [`GaussLobatto`](@ref), [`QuadraturePoint`](@ref)
"""
abstract type AbstractQuadratureRule end

"""
    GaussLegendre{N} <: AbstractQuadratureRule

Gauss-Legendre quadrature with N points per dimension.

Gauss-Legendre quadrature is optimal for polynomial integration: N points integrate
polynomials of degree 2N-1 exactly. These are "open" rules that do not include
the element boundaries, making them ideal for smooth integrands.

# Type Parameter
- `N::Int`: Number of quadrature points per dimension

# Integration Accuracy
- 1D line: N points integrates degree 2N-1 polynomials exactly
- 2D quad: N² points (tensor product)
- 2D triangle: Variable points (not tensor product)
- 3D hex: N³ points (tensor product)
- 3D tetrahedron: Variable points (not tensor product)

# Examples

    # 1-point quadrature (exact for linear)
    GaussLegendre{1}()

    # 2-point quadrature (exact for cubic)
    GaussLegendre{2}()

    # 3-point quadrature (exact for quintic)
    GaussLegendre{3}()

    # Get quadrature points for specific topology
    points = get_quadrature_points(Triangle, GaussLegendre{2}())
    # Returns SVector of QuadraturePoint{2}

# References
- Abramowitz & Stegun, "Handbook of Mathematical Functions"
- Dunavant, "High degree efficient symmetrical Gaussian quadrature rules for the triangle"

See also: [`AbstractQuadratureRule`](@ref), [`GaussLobatto`](@ref), [`QuadraturePoint`](@ref)
"""
struct GaussLegendre{N,V} <: AbstractQuadratureRule end

# Constructor with default variant parameter
GaussLegendre{N}() where {N} = GaussLegendre{N,:default}()

"""
    GaussLobatto{N} <: AbstractQuadratureRule

Gauss-Lobatto quadrature with N points per dimension.

Gauss-Lobatto quadrature includes the element boundary points, making it useful
for spectral element methods and discontinuous Galerkin methods where boundary
values are important. N points integrate polynomials of degree 2N-3 exactly
(slightly lower accuracy than Gauss-Legendre).

# Type Parameter
- `N::Int`: Number of quadrature points per dimension (N ≥ 2)

# Integration Accuracy
- 1D line: N points integrates degree 2N-3 polynomials exactly
- Higher dimensions: Tensor products follow same rule

# Key Differences from Gauss-Legendre
- **Includes boundaries**: First and last points are at ±1
- **Lower accuracy**: Degree 2N-3 vs 2N-1 for same number of points
- **Useful for**: Spectral methods, DG methods, boundary coupling

# Examples
```julia
# 2-point Lobatto (just endpoints, degree 1)
GaussLobatto{2}()

# 3-point Lobatto (includes midpoint, degree 3)
GaussLobatto{3}()

# Get quadrature points for quadrilateral
points = get_quadrature_points(Quadrilateral, GaussLobatto{3}())
```

# Note
Not all topologies support Gauss-Lobatto rules. Currently implemented for
tensor-product elements (segments, quadrilaterals, hexahedra).

See also: [`AbstractQuadratureRule`](@ref), [`GaussLegendre`](@ref)
"""
struct GaussLobatto{N,V} <: AbstractQuadratureRule end

# Constructor with default variant parameter
GaussLobatto{N}() where {N} = GaussLobatto{N,:default}()

"""
    QuadraturePoint{D,T<:Real}

Represents a single quadrature (integration) point in D-dimensional parametric space.

This is a lightweight struct containing the parametric coordinates and integration
weight for a single quadrature point. Designed for zero-allocation when used in
SVector containers.

# Type Parameters
- `D::Int`: Spatial dimension (1D, 2D, or 3D)
- `T::Real`: Numeric type for coordinates and weight (typically Float64)

# Fields
- `coords::Vec{D,T}`: Location in parametric coordinates (ξ, η, ζ)
- `weight::T`: Integration weight

# Examples
```julia
# 2D quadrature point at triangle centroid
qp = QuadraturePoint(Vec{2}(1/3, 1/3), 0.5)

# Access components
ξ = qp.coords[1]    # First parametric coordinate
η = qp.coords[2]    # Second parametric coordinate
w = qp.weight       # Integration weight

# Usage in assembly loop
for qp in quadrature_points
    N = evaluate_basis(qp.coords)
    detJ = compute_jacobian(qp.coords)
    K += qp.weight * detJ * (B' * D * B)
end
```

# Performance Notes
- Immutable struct → stack allocated
- Vec → SIMD-friendly, compatible with Tensors.jl
- Small enough to pass by value efficiently
- Zero allocation when used in SVector containers

See also: [`AbstractQuadratureRule`](@ref), [`GaussLegendre`](@ref)
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

Return the quadrature points and weights for the given topology and quadrature rule.

**Zero allocation:** Returns compile-time sized SVector of QuadraturePoints for
all standard rules. This function is the primary interface for obtaining quadrature
data in assembly loops.

# Arguments
- `topology`: Topology type (e.g., `Triangle`, `Tetrahedron`, `Hexahedron`)
- `rule`: Quadrature rule (e.g., `GaussLegendre{2}()`)

# Returns
SVector of `QuadraturePoint{D}` where D is the dimension of the topology.

# Examples
```julia
# Get 3-point Gauss rule for triangle
points = get_quadrature_points(Triangle, GaussLegendre{2}())
# Returns: SVector{3, QuadraturePoint{2,Float64}}

# Usage in assembly
for qp in get_quadrature_points(Hexahedron, GaussLegendre{2}())
    # qp.coords → Vec{3,Float64} with (ξ, η, ζ)
    # qp.weight → Float64
    N = evaluate_basis(qp.coords)
    # ...
end
```

# Performance
- Zero allocations (fully compile-time resolved)
- Type-stable (return type known at compile time)
- SIMD-friendly (SVector operations)
- Inlined for maximum performance

See also: [`QuadraturePoint`](@ref), [`GaussLegendre`](@ref)
"""
function get_quadrature_points end

"""
    npoints(topology::Type{<:AbstractTopology}, rule::AbstractQuadratureRule) -> Int

Return the number of quadrature points for the given topology and rule.

# Examples
```julia
julia> npoints(Triangle, GaussLegendre{2}())
3

julia> npoints(Hexahedron, GaussLegendre{2}())
8
```
"""
npoints(topology::Type{<:AbstractTopology}, rule::AbstractQuadratureRule) =
    length(get_quadrature_points(topology, rule))

"""
    default_quadrature(basis_order::Int) -> AbstractQuadratureRule
    default_quadrature(topology::Type{<:AbstractTopology}, basis_order::Int) -> AbstractQuadratureRule
    default_quadrature(topology::Type{<:AbstractTopology}, basis::Type{<:AbstractBasis}) -> AbstractQuadratureRule
    default_quadrature(topology::Type{<:AbstractTopology}) -> AbstractQuadratureRule

Return the default (recommended) quadrature rule for given topology and/or basis.

This function provides multi-level dispatch for automatic quadrature selection:

# Dispatch Levels

**Level 1: From basis order only**
```julia
default_quadrature(1)  # → GaussLegendre{2}() for linear basis
default_quadrature(2)  # → GaussLegendre{3}() for quadratic basis
```

**Level 2: From topology + basis order**
```julia
default_quadrature(Hexahedron, 1)  # → GaussLegendre{2}() (2×2×2 = 8 pts)
default_quadrature(Triangle, 2)    # → GaussLegendre{3}() (4 pts)
```

**Level 3: From topology + basis type**
```julia
default_quadrature(Hexahedron{8}, Lagrange{1})   # → GaussLegendre{2}()
default_quadrature(Hexahedron{27}, Lagrange{2})  # → GaussLegendre{3}()
```

**Level 4: From topology type only (infers basis from node count)**
```julia
default_quadrature(Hexahedron{8})   # → GaussLegendre{2}() (infers linear)
default_quadrature(Hexahedron{27})  # → GaussLegendre{3}() (infers quadratic)
default_quadrature(Triangle{3})     # → GaussLegendre{2}() (infers linear)
```

# Default Rules

Uses `GaussLegendre{basis_order + 1}()` to ensure exact integration of
stiffness matrices where grad(Ni)·grad(Nj) must be integrated.

- Linear basis (order 1): GaussLegendre{2}()
- Quadratic basis (order 2): GaussLegendre{3}()
- Cubic basis (order 3): GaussLegendre{4}()

# Examples
```julia
# Automatic from topology
julia> default_quadrature(Hexahedron{8})
GaussLegendre{2}()  # 2×2×2 = 8 points (exact for trilinear)

# Automatic from topology + basis
julia> rule = default_quadrature(Triangle{6}, Lagrange{2})
GaussLegendre{3}()  # 4 points (exact for quadratic)

julia> points = get_quadrature_points(Triangle, rule)
SVector{4, QuadraturePoint{2,Float64}}
```

See also: [`get_quadrature_points`](@ref), [`GaussLegendre`](@ref)
"""
function default_quadrature end

# Level 1: Just basis order (simplest, works for most cases)
# Rule of thumb: Use order + 1 for stiffness matrix integration
default_quadrature(basis_order::Int) = GaussLegendre{basis_order + 1}()

# Level 2: Topology + basis order (handles special cases, can be overridden)
# Default implementation delegates to Level 1
default_quadrature(::Type{<:AbstractTopology}, basis_order::Int) =
    default_quadrature(basis_order)

# Level 3: Topology + basis type (extract order from basis)
# Note: Requires basis_order(::Type{<:AbstractBasis}) to be defined
# This will be implemented when basis types are loaded
# default_quadrature(::Type{T}, ::Type{B}) where {T<:AbstractTopology, B<:AbstractBasis} =
#     default_quadrature(T, basis_order(B))

# Level 4: Topology type only (infer basis order from node count)
# These will be uncommented after topology types are loaded

# Helper function to infer basis order from node count
# This is a heuristic based on standard element types
function _infer_basis_order end

# 1D Segments
# _infer_basis_order(::Type{Segment{2}}) = 1   # Linear
# _infer_basis_order(::Type{Segment{3}}) = 2   # Quadratic

# 2D Triangles
# _infer_basis_order(::Type{Triangle{3}}) = 1   # Linear
# _infer_basis_order(::Type{Triangle{6}}) = 2   # Quadratic
# _infer_basis_order(::Type{Triangle{7}}) = 2   # Quadratic with center

# 2D Quadrilaterals
# _infer_basis_order(::Type{Quadrilateral{4}}) = 1   # Bilinear
# _infer_basis_order(::Type{Quadrilateral{8}}) = 2   # Serendipity
# _infer_basis_order(::Type{Quadrilateral{9}}) = 2   # Biquadratic

# 3D Tetrahedra
# _infer_basis_order(::Type{Tetrahedron{4}}) = 1    # Linear
# _infer_basis_order(::Type{Tetrahedron{10}}) = 2   # Quadratic

# 3D Hexahedra
# _infer_basis_order(::Type{Hexahedron{8}}) = 1     # Trilinear
# _infer_basis_order(::Type{Hexahedron{20}}) = 2    # Serendipity
# _infer_basis_order(::Type{Hexahedron{27}}) = 2    # Triquadratic

# 3D Wedges
# _infer_basis_order(::Type{Wedge{6}}) = 1          # Linear
# _infer_basis_order(::Type{Wedge{15}}) = 2         # Quadratic

# 3D Pyramids
# _infer_basis_order(::Type{Pyramid{5}}) = 1        # Linear

# Topology-only dispatch (uses inferred basis order)
# default_quadrature(::Type{T}) where {T<:AbstractTopology} =
#     default_quadrature(T, _infer_basis_order(T))
