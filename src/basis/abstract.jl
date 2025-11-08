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
eval_basis!(B::T, N, xi) where {T<:AbstractBasis} = eval_basis!(T, N, xi)
eval_dbasis!(B::T, dN, xi) where {T<:AbstractBasis} = eval_dbasis!(T, dN, xi)

# Allocating versions (convenience wrappers)
"""
    eval_basis(basis::AbstractBasis{dim}, xi) -> Vector{Float64}

Evaluate basis functions at point `xi`, allocating return vector.

See also: [`eval_basis!`](@ref) for non-allocating version.
"""
eval_basis(B::AbstractBasis{dim}, xi) where {dim} = eval_basis!(B, zeros(length(B)), xi)

"""
    eval_dbasis(basis::AbstractBasis{dim}, xi) -> Vector{Vec{dim, Float64}}

Evaluate basis function derivatives at point `xi`, allocating return vector.

See also: [`eval_dbasis!`](@ref) for non-allocating version.
"""
eval_dbasis(B::AbstractBasis{dim}, xi) where {dim} = eval_dbasis!(B, zeros(Vec{dim}, length(B)), xi)

# Declare interface functions (will be implemented by basis generator)
function get_reference_element_coordinates end
function eval_basis! end
function eval_dbasis! end
