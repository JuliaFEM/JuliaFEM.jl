# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE

# AbstractBasis type and interface
# Consolidated from jl package

using Tensors
using LinearAlgebra
# import Calculus  # Only needed for symbolic basis generation (create_basis.jl)

# Re-export Vec for convenience (from Tensors.jl)
export Vec

# Type alias for coordinate inputs (tuples or Vec)
const Vecish{N,T} = Union{NTuple{N,T},Vec{N,T}}

"""
    AbstractBasis

Abstract base type for all finite element basis functions.

# Interface requirements

Concrete basis types must implement:
- `nnodes(::Type{<:AbstractBasis})` - Number of basis functions
- `Base.ndims(::Type{<:AbstractBasis})` - Spatial dimension
- `eval_basis!(::Type{<:AbstractBasis}, N, xi)` - Evaluate basis functions
- `eval_dbasis!(::Type{<:AbstractBasis}, dN, xi)` - Evaluate basis derivatives

# Example

```julia
struct Lagrange{Triangle,1} <: AbstractBasis end
nnodes(Lagrange{Triangle,1}) == 3
ndims(Lagrange{Triangle,1}) == 2
```

See also: [`Lagrange`](@ref), [`Serendipity`](@ref)
"""
abstract type AbstractBasis end

# Forward methods on instances to types
# This allows calling methods on both Lagrange{Triangle,1} and Lagrange{Triangle,1}()
Base.ndims(B::T) where {T<:AbstractBasis} = ndims(T)
nnodes(B::T) where {T<:AbstractBasis} = nnodes(T)

# Declare interface functions (will be implemented by basis generator or specific basis types)
function eval_basis! end
function eval_dbasis! end

# New API functions (declared here, implemented in basis_api.jl)
function get_basis_functions end
function get_basis_derivatives end

# ============================================================================
# Parametric Lagrange Basis Type (OLD - with topology parameter)
# ============================================================================
# NOTE: This is the OLD API with topology in the type parameter.
# New code should use LagrangeP{P} below (topology passed separately).
# This is kept for backward compatibility during migration.

"""
    Lagrange{T<:AbstractTopology, P} <: AbstractBasis

Parametric Lagrange basis functions for topology `T` with polynomial degree `P`.

The node count is automatically derived from the topology and polynomial degree:
- `Lagrange{Triangle, 1}`: P1 → 3 nodes (vertices)
- `Lagrange{Triangle, 2}`: P2 → 6 nodes (vertices + edge midpoints)
- `Lagrange{Quadrilateral, 1}`: Q1 → 4 nodes (corners)
- `Lagrange{Quadrilateral, 2}`: Q2 → 9 nodes (full tensor product)
- `Lagrange{Tetrahedron, 1}`: P1 → 4 nodes (vertices)
- `Lagrange{Tetrahedron, 2}`: P2 → 10 nodes (vertices + edge midpoints)
- `Lagrange{Hexahedron, 1}`: Q1 → 8 nodes (corners)
- `Lagrange{Hexahedron, 2}`: Q2 → 27 nodes (full tensor product)

# Type Parameters
- `T`: Topology type (Triangle, Quadrilateral, Tetrahedron, Hexahedron, etc.)
- `P`: Polynomial degree (1 = linear, 2 = quadratic, 3 = cubic, ...)

# Mathematical Background

Lagrange basis functions satisfy the cardinal property:
```
Nᵢ(xⱼ) = δᵢⱼ  (Kronecker delta)
```

where `xⱼ` are the interpolation nodes.

For polynomial degree P:
- 1D: P+1 nodes
- Triangle: (P+1)(P+2)/2 nodes
- Quadrilateral: (P+1)² nodes (tensor product)
- Tetrahedron: (P+1)(P+2)(P+3)/6 nodes
- Hexahedron: (P+1)³ nodes (tensor product)

# Example

```julia
# Linear triangle (P1)
basis = Lagrange{Triangle, 1}()
nnodes(basis)  # → 3

# Quadratic triangle (P2)
basis = Lagrange{Triangle, 2}()
nnodes(basis)  # → 6

# Bilinear quadrilateral (Q1)
basis = Lagrange{Quadrilateral, 1}()
nnodes(basis)  # → 4

# Biquadratic quadrilateral (Q2)
basis = Lagrange{Quadrilateral, 2}()
nnodes(basis)  # → 9
```

See also: [`AbstractBasis`](@ref), [`Serendipity`](@ref), [`Nedelec`](@ref)
"""
struct Lagrange{T<:AbstractTopology,P} <: AbstractBasis end

# Define interface methods for Lagrange
# Dimension comes from topology
Base.ndims(::Type{Lagrange{T,P}}) where {T,P} = dim(T())
Base.ndims(::Lagrange{T,P}) where {T,P} = dim(T())

# Node count formulas for different topologies and polynomial degrees
# These replace the hardcoded node counts in old Tri3, Quad4, etc. types

"""
    nnodes(::Lagrange{T, P}) where {T, P}
    nnodes(::Type{Lagrange{T, P}}) where {T, P}

Compute number of nodes for Lagrange basis of degree P on topology T.
Works with both instances and types.
"""
# 1D: Segment
nnodes(::Lagrange{Segment,P}) where {P} = P + 1
nnodes(::Type{Lagrange{Segment,P}}) where {P} = P + 1

# 2D: Triangle (simplex)
nnodes(::Lagrange{Triangle,P}) where {P} = div((P + 1) * (P + 2), 2)
nnodes(::Type{Lagrange{Triangle,P}}) where {P} = div((P + 1) * (P + 2), 2)

# 2D: Quadrilateral (tensor product)
nnodes(::Lagrange{Quadrilateral,P}) where {P} = (P + 1)^2
nnodes(::Type{Lagrange{Quadrilateral,P}}) where {P} = (P + 1)^2

# 3D: Tetrahedron (simplex)
nnodes(::Lagrange{Tetrahedron,P}) where {P} = div((P + 1) * (P + 2) * (P + 3), 6)
nnodes(::Type{Lagrange{Tetrahedron,P}}) where {P} = div((P + 1) * (P + 2) * (P + 3), 6)

# 3D: Hexahedron (tensor product)
nnodes(::Lagrange{Hexahedron,P}) where {P} = (P + 1)^3
nnodes(::Type{Lagrange{Hexahedron,P}}) where {P} = (P + 1)^3

# 3D: Pyramid (mixed)
# Pyramids don't follow a simple formula, so hardcode for known degrees
nnodes(::Lagrange{Pyramid,1}) = 5
nnodes(::Type{Lagrange{Pyramid,1}}) = 5
nnodes(::Lagrange{Pyramid,2}) = 13
nnodes(::Type{Lagrange{Pyramid,2}}) = 13
nnodes(::Lagrange{Pyramid,3}) = 29
nnodes(::Type{Lagrange{Pyramid,3}}) = 29

# 3D: Wedge/Prism (triangle × segment tensor product)
nnodes(::Lagrange{Wedge,P}) where {P} = div((P + 1)^2 * (P + 2), 2)
nnodes(::Type{Lagrange{Wedge,P}}) where {P} = div((P + 1)^2 * (P + 2), 2)

# Also need nnodes for the higher-order topology types themselves (Tet10, Tri6, etc.)
# These forward to the topology's nnodes() method
nnodes(::Type{T}) where {T<:AbstractTopology} = nnodes(T())

# For backwards compatibility, support old topology type names as if they were basis types
# This handles cases where code uses Tri3, Quad4, etc. as basis types
nnodes(::Type{Tri3}) = 3
nnodes(::Type{Tri6}) = 6
nnodes(::Type{Tri7}) = 7
nnodes(::Type{Quad4}) = 4
nnodes(::Type{Quad8}) = 8
nnodes(::Type{Quad9}) = 9
nnodes(::Type{Seg2}) = 2
nnodes(::Type{Seg3}) = 3
nnodes(::Type{Tet4}) = 4
nnodes(::Type{Tet10}) = 10
nnodes(::Type{Hex8}) = 8
nnodes(::Type{Hex20}) = 20
nnodes(::Type{Hex27}) = 27
nnodes(::Type{Pyr5}) = 5
nnodes(::Type{Wedge6}) = 6
nnodes(::Type{Wedge15}) = 15

# Export the new parametric type and node count function
export Lagrange, nnodes
