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
