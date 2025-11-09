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
    AbstractBasis{dim}

Abstract base type for all finite element basis functions.

# Type parameter
- `dim`: Dimensionality of the reference element (1, 2, or 3)

# Interface requirements

Concrete basis types must implement:
- `Base.length(::Type{<:AbstractBasis})` - Number of basis functions
- `Base.size(::Type{<:AbstractBasis})` - (dim, n_basis)
- `get_reference_element_coordinates(::Type{<:AbstractBasis})` - Reference coordinates
- `eval_basis!(::Type{<:AbstractBasis}, N, xi)` - Evaluate basis functions
- `eval_dbasis!(::Type{<:AbstractBasis}, dN, xi)` - Evaluate basis derivatives

# Example

```julia
struct Seg2 <: AbstractBasis{1} end
length(Seg2) == 2
size(Seg2) == (1, 2)
```
"""
abstract type AbstractBasis{dim} end

# Forward methods on instances to types
# This allows calling methods on both Seg2 and Seg2()
Base.length(B::T) where {T<:AbstractBasis} = length(T)
Base.size(B::T) where {T<:AbstractBasis} = size(T)
# Updated signatures: eval_basis! and eval_dbasis! now return tuples
eval_basis!(B::T, ::Type{U}, xi) where {T<:AbstractBasis,U} = eval_basis!(T, U, xi)
eval_dbasis!(B::T, xi) where {T<:AbstractBasis} = eval_dbasis!(T, xi)

# Allocating versions (convenience wrappers) - now they just call and collect
"""
    eval_basis(basis::AbstractBasis{dim}, xi) -> Vector{Float64}

Evaluate basis functions at point `xi`, allocating return vector.

See also: [`eval_basis!`](@ref) for non-allocating version that returns tuple.
"""
eval_basis(B::AbstractBasis{dim}, ::Type{T}, xi) where {dim,T} = collect(eval_basis!(B, T, xi))

"""
    eval_dbasis(basis::AbstractBasis{dim}, xi) -> Vector{Vec{dim, Float64}}

Evaluate basis function derivatives at point `xi`, allocating return vector.

See also: [`eval_dbasis!`](@ref) for non-allocating version that returns tuple.
"""
eval_dbasis(B::AbstractBasis{dim}, xi) where {dim} = collect(eval_dbasis!(B, xi))

# Declare interface functions (will be implemented by basis generator)
function get_reference_element_coordinates end
function eval_basis! end
function eval_dbasis! end

# ============================================================================
# Parametric Lagrange Basis Type
# ============================================================================

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
# Lagrange inherits from AbstractBasis but dimension is determined by topology
# We cannot compute dim(T()) at type definition time, so we use methods instead
struct Lagrange{T<:AbstractTopology,P} end

# Define interface methods for Lagrange
# Dimension comes from topology
Base.ndims(::Type{Lagrange{T,P}}) where {T,P} = dim(T())
Base.ndims(::Lagrange{T,P}) where {T,P} = dim(T())

# Node count formulas for different topologies and polynomial degrees
# These replace the hardcoded node counts in old Tri3, Quad4, etc. types

"""
    nnodes(::Lagrange{T, P}) where {T, P}

Compute number of nodes for Lagrange basis of degree P on topology T.
"""
# 1D: Segment
nnodes(::Lagrange{Segment,P}) where {P} = P + 1

# 2D: Triangle (simplex)
nnodes(::Lagrange{Triangle,P}) where {P} = div((P + 1) * (P + 2), 2)

# 2D: Quadrilateral (tensor product)
nnodes(::Lagrange{Quadrilateral,P}) where {P} = (P + 1)^2

# 3D: Tetrahedron (simplex)
nnodes(::Lagrange{Tetrahedron,P}) where {P} = div((P + 1) * (P + 2) * (P + 3), 6)

# 3D: Hexahedron (tensor product)
nnodes(::Lagrange{Hexahedron,P}) where {P} = (P + 1)^3

# 3D: Pyramid (mixed)
# Pyramids don't follow a simple formula, so hardcode for known degrees
nnodes(::Lagrange{Pyramid,1}) = 5
nnodes(::Lagrange{Pyramid,2}) = 13
nnodes(::Lagrange{Pyramid,3}) = 29

# 3D: Wedge/Prism (triangle × segment tensor product)
nnodes(::Lagrange{Wedge,P}) where {P} = div((P + 1)^2 * (P + 2), 2)

# Export the new parametric type and node count function
export Lagrange, nnodes
