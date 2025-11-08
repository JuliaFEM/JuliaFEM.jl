# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/jl/blob/master/LICENSE

"""
    interpolate(B, T, xi)

Given basis B, interpolate T at xi.

# Example
```jldoctest
B = Quad4()
X = Vec.([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
T = [1.0, 2.0, 3.0, 4.0]
interpolate(B, T, Vec(0.0, 0.0))

# output

2.5
```
"""
function interpolate(B::AbstractBasis{dim}, T::Vector, xi::Vec{dim}) where {dim}
    N = eval_basis(B, xi)
    return sum(b*t for (b, t) in zip(N, T))
end

"""
    jacobian(B, X, xi)

Given basis B, calculate jacobian at xi.

# Example
```jldoctest
B = Quad4()
X = Vec.([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
jacobian(B, X, Vec((0.0, 0.0)))

# output

2×2 Tensor{2,2,Float64,4}:
 0.5  0.0
 0.0  0.5

```
"""
jacobian(B::AbstractBasis{dim}, X::Vector{<:Vec{dim}}, xi::Vec{dim}) where {dim} = jacobian(B, X, xi, eval_dbasis(B, xi))

function jacobian(B::AbstractBasis{dim}, X::Vector{<:Vec{dim}}, xi::Vec{dim}, dB::Vector{<:Vec{dim}}) where {dim}
    @assert length(X) == length(B) == length(dB)
    J = zero(Tensor{2, dim})
    @inbounds for i in 1:length(X)
        J += otimes(dB[i], X[i]) # dB[i] ⊗ X[i]
    end
    return J
end



"""
    grad(B, X, xi)

Given basis B, calculate gradient dB/dX at xi.

# Example
```jldoctest
B = Quad4()
X = Vec.([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
grad(B, X, Vec(0.0, 0.0))

# output

4-element Array{Tensor{1,2,Float64,2},1}:
 [-0.5, -0.5]
 [0.5, -0.5]
 [0.5, 0.5]
 [-0.5, 0.5]

```
"""
grad(B::AbstractBasis{dim}, X::Vector{<:Vec{dim}}, xi::Vec{dim}) where {dim} =
    grad!(B, similar(X), X, xi, eval_dbasis(B, xi))

function grad!(B::AbstractBasis{dim}, dN::Vector{<:Vec{dim}}, X::Vector{<:Vec{dim}}, xi::Vec{dim}, dB::Vector{<:Vec{dim}}) where {dim}
    @assert length(dN) == length(dB)
    J = jacobian(B, X, xi, dB)
    @inbounds for i in 1:length(dN)
        dN[i] = inv(J) ⋅ dB[i]
    end
    return dN
end

"""
    grad(B, T, X, xi)

Calculate gradient of `T` with respect to `X` in point `xi` using basis `B`.

# Example
```jldoctest
B = Quad4()
X = Vec.([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
u = Vec.([(0.0, 0.0), (1.0, -1.0), (2.0, 3.0), (0.0, 0.0)])
grad(B, u, X, Vec(0.0, 0.0))

# output

julia> grad(B, u, X, Vec(0.0, 0.0))
2×2 Tensor{2,2,Float64,4}:
 1.5  0.5
 1.0  2.0

```
"""
function grad(B::AbstractBasis{dim}, T::Vector{<:Vec{dim}}, X::Vector{<:Vec{dim}}, xi::Vec{dim}) where {dim}
    G = grad(B, X, xi) # <- allocates
    dTdX = sum(T[i] ⊗ G[i] for i=1:length(B))
    return dTdX
end
function grad(B::AbstractBasis{dim}, T::Vector{<:Number}, X::Vector{<:Vec{dim}}, xi::Vec{dim}) where {dim}
    G = grad(B, X, xi) # <- allocates
    dTdX = sum(T[i] * G[i] for i=1:length(B))
    return dTdX
end


"""
Data type for fast FEM.
"""
mutable struct BasisInfo{B<:AbstractBasis,dim, T, M}
    N::Vector{T}
    dN::Vector{Vec{dim, T}}
    grad::Vector{Vec{dim, T}}
    J::Tensor{2, dim, T, M}
    invJ::Tensor{2, dim, T, M}
    detJ::T
    basis::Type{B}
end

Base.length(B::BasisInfo{T}) where T<:AbstractBasis = length(T)
Base.size(B::BasisInfo{T})   where T<:AbstractBasis = size(T)

"""
Initialization of data type `BasisInfo`.

# Examples

```jldoctest

BasisInfo(Tri3)

# output

BasisInfo{Tri3,Float64}([0.0 0.0 0.0], [0.0 0.0 0.0; 0.0 0.0 0.0], [0.0 0.0 0.0; 0.0 0.0 0.0], [0.0 0.0; 0.0 0.0], [0.0 0.0; 0.0 0.0], 0.0)

```

"""
function BasisInfo(::Type{B}, T=Float64) where B <: AbstractBasis{dim} where dim
    nbasis = length(B)
    N = zeros(T, nbasis)
    dN = zeros(Vec{dim, T}, nbasis)
    grad = zeros(Vec{dim, T}, nbasis)
    J = zero(Tensor{2, dim, T})
    invJ = zero(Tensor{2, dim, T})
    detJ = zero(T)
    return BasisInfo(N, dN, grad, J, invJ, detJ, B)
end

"""
Evaluate basis, gradient and so on for some point `xi`.

# Examples

```jldoctest

b = BasisInfo(Quad4)
X = Vec.([(0.0,0.0), (1.0,0.0), (1.0,1.0), (0.0,1.0)])
xi = Vec(0.0, 0.0)
eval_basis!(b, X, xi)

# output

BasisInfo{Quad4,2,Float64,4}([0.25, 0.25, 0.25, 0.25], Tensors.Tensor{1,2,Float64,2}[[-0.25, -0.25], [0.25, -0.25], [0.25, 0.25], [-0.25, 0.25]], Tensors.Tensor{1,2,Float64,2}[[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]], [0.5 0.0; 0.0 0.5], [2.0 -0.0; -0.0 2.0], 0.25, Quad4)

```
"""
function eval_basis!(bi::BasisInfo{B},
                     X::Vector{<:Vec{dim}}, xi::Vec{dim}) where B <: AbstractBasis{dim} where dim
    # evaluate basis and derivatives
    eval_basis!(B, bi.N, xi)
    eval_dbasis!(B, bi.dN, xi)

    # calculate Jacobian
    bi.J = jacobian(B(), X, xi, bi.dN)

    # calculate determinant of Jacobian + gradient operator

    # TODO, fixup curve + manifold
 #   @assert dim[1] == dim[2]
    bi.invJ = inv(bi.J)
    @inbounds for i in 1:length(bi.dN)
        bi.grad[i] = bi.invJ ⋅ bi.dN[i]
    end
    bi.detJ = det(bi.J)
    #=
    elseif dim1 == 1 # curve
        bi.detJ = norm(bi.J)
    elseif dim1 == 2 # manifold
        bi.detJ = norm(cross(bi.J[1,:], bi.J[2,:]))
    end
    =#

    return bi
end

"""
    grad!(bi, gradu, u)

Evalute gradient ∂u/∂X and store result to matrix `gradu`. It is assumed
that `eval_basis!` has been already run to `bi` so it already contains
all necessary matrices evaluated with some `X` and `xi`.

# Example

First setup and evaluate basis using `eval_basis!`:
```jldoctest ex1
B = BasisInfo(Quad4)
X = Vec.([(0.0,0.0), (1.0,0.0), (1.0,1.0), (0.0,1.0)])
xi = Vec(0.0, 0.0)
eval_basis!(B, X, xi)

# output

BasisInfo{Quad4,2,Float64}([0.25, 0.25, 0.25, 0.25], Tensors.Tensor{1,2,Float64,2}[[-0.25, -0.25], [0.25, -0.25], [0.25, 0.25], [-0.25, 0.25]], Tensors.Tensor{1,2,Float64,2}[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [-0.5, 0.5]], [0.5 0.0; 0.0 0.5], [2.0 -0.0; -0.0 2.0], 0.25)

```

Next, calculate gradient of `u`:
```jldoctest ex1
u = Vec.([(0.0, 0.0), (1.0, -1.0), (2.0, 3.0), (0.0, 0.0)])
grad(B, u)

# output

2×2 Tensors.Tensor{2,2,Float64,4}:
 1.5  0.5
 1.0  2.0

```
"""
function grad(bi::BasisInfo{B}, u::Vector{<:Vec{dim}}) where B <: AbstractBasis{dim} where dim
    gradu = zero(Tensor{2, dim})
    for k in 1:length(B)
        gradu += otimes(u[k], bi.grad[k])
    end
    return gradu
end
