# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/FEMBase.jl/blob/master/LICENSE
#
# Sparse matrix utilities consolidated from FEMBase.jl and FEMSparse.jl

using SparseArrays
import SparseArrays: sparse, sparsevec

# SparseMatrixCOO from FEMBase (this file)
# AssemblerSparsityPattern from FEMSparse
include("sparsematrixcsc.jl")
# include("sparsevectordok.jl")  # Old Julia syntax, not used, skipping for now

mutable struct SparseMatrixCOO{T<:Real}
    I :: Vector{Int}
    J :: Vector{Int}
    V :: Vector{T}
end

const SparseVectorCOO = SparseMatrixCOO

function SparseMatrixCOO()
    return SparseMatrixCOO{Float64}([], [], [])
end

function SparseVectorCOO(I::Vector, V::Vector)
    return SparseVectorCOO(I, fill!(similar(I), 1), V)
end

function convert(::Type{SparseMatrixCOO}, A::SparseMatrixCSC)
    return SparseMatrixCOO(findnz(A)...)
end

function convert(::Type{SparseVectorCOO}, A::SparseVector)
    return SparseVectorCOO(findnz(A)...)
end

function convert(::Type{SparseMatrixCOO}, A::Matrix)
    idx = findall(!iszero, A)
    I = getindex.(idx, 1)
    J = getindex.(idx, 2)
    V = [A[i] for i in idx]
    return SparseMatrixCOO(I, J, V)
end

function convert(::Type{SparseMatrixCOO}, b::Vector)
    I = findall(!iszero, b)
    J = fill(1, size(I))
    V = b[I]
    return SparseMatrixCOO(I, J, V)
end

SparseArrays.sparse(A::SparseMatrixCOO) = sparse(A.I, A.J, A.V)
SparseArrays.sparse(A::SparseMatrixCOO, n::Int, m::Int) = sparse(A.I, A.J, A.V, n, m)
SparseArrays.sparse(A::SparseMatrixCOO, n::Int, m::Int, f::Function) = sparse(A.I, A.J, A.V, n, m, f)
Base.Matrix(A::SparseMatrixCOO) = Matrix(sparse(A))
Base.Matrix(A::SparseMatrixCOO, n::Int, m::Int) = Matrix(sparse(A, n, m))

SparseArrays.sparsevec(b::SparseVectorCOO) = sparsevec(b.I, b.V)
SparseArrays.sparsevec(b::SparseVectorCOO, n::Int) = sparsevec(b.I, b.V, n)
Base.Vector(b::SparseVectorCOO) = Vector(sparsevec(b))
Base.Vector(b::SparseVectorCOO, n::Int) = Vector(sparsevec(b, n))

function add!(A::SparseMatrixCOO, I::Int, J::Int, V::Float64)
    push!(A.I, I)
    push!(A.J, J)
    push!(A.V, V)
    return nothing
end

function add!(A::SparseMatrixCOO, I::Int, V::Float64)
    push!(A.I, I)
    push!(A.J, 1)
    push!(A.V, V)
    return nothing
end

function empty!(A::SparseMatrixCOO)
    empty!(A.I)
    empty!(A.J)
    empty!(A.V)
    return nothing
end

function append!(A::SparseMatrixCOO, B::SparseMatrixCOO)
    append!(A.I, B.I)
    append!(A.J, B.J)
    append!(A.V, B.V)
    return nothing
end

function isempty(A::SparseMatrixCOO)
    return isempty(A.I) && isempty(A.J) && isempty(A.V)
end

"""
    add!(K, dofs1, dofs2, ke)

Add local element matrix `ke` to sparse matrix `K` for indices defined by `dofs1`
and `dofs2`. This basically does `A[dofs1, dofs2] = A[dofs1, dofs2] + data`.

# Examples

```julia
S = [3, 4]
M = [6, 7, 8]
ke = [5 6 7; 8 9 10]
K = SparseMatrixCOO()
add!(K, S, M, ke)
Matrix(A)

# output

4x8 Array{Float64,2}:
 0.0  0.0  0.0  0.0  0.0  0.0  0.0   0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0   0.0
 0.0  0.0  0.0  0.0  0.0  5.0  6.0   7.0
 0.0  0.0  0.0  0.0  0.0  8.0  9.0  10.0
```
"""
function add!(A::SparseMatrixCOO, dofs1::AbstractVector{Int}, dofs2::AbstractVector{Int}, data)
    n, m = length(dofs1), length(dofs2)
    @assert length(data) == n*m
    k = 1
    for j=1:m
        for i=1:n
            add!(A, dofs1[i], dofs2[j], data[k])
            k += 1
        end
    end
    return nothing
end

""" Add sparse matrix of CSC to COO. """
function add!(A::SparseMatrixCOO, B::SparseMatrixCSC)
    i, j, v = findnz(B)
    C = SparseMatrixCOO(i, j, v)
    append!(A, C)
end

""" Add new data to COO Sparse vector. """
function add!(A::SparseMatrixCOO, dofs::Vector{Int}, data::Array{Float64}, dim::Int=1)
    if length(dofs) != length(data)
        @error("Dimension mismatch when adding data to sparse vector!", dofs, data)
        error("Simulation stopped.")
    end
    append!(A.I, dofs)
    append!(A.J, dim*ones(Int, length(dofs)))
    append!(A.V, vec(data))
end

""" Add SparseVector to SparseVectorCOO. """
function add!(a::SparseVectorCOO, b::SparseVector)
    i, v = findnz(b)
    c = SparseVectorCOO(i, v)
    append!(a, c)
    return
end

"""
    get_nonzero_rows(A)

Returns indices of all nonzero rows from a sparse matrix `A`.
"""
function get_nonzero_rows(A)
    return sort(unique(A.rowval))
end

"""
    get_nonzero_columns(A)

Returns indices of all nonzero columns from a sparse matrix `A`.
"""
function get_nonzero_columns(A)
    return get_nonzero_rows(copy(transpose(A)))
end

function size(A::SparseMatrixCOO)
    isempty(A) && return (0, 0)
    return maximum(A.I), maximum(A.J)
end

function size(A::SparseMatrixCOO, idx::Int)
    return size(A)[idx]
end

""" Resize sparse matrix A to (higher) dimension n x m. """
function resize_sparse(A, n, m)
    idx = findall(!iszero, A)
    I = getindex.(idx, 1)
    J = getindex.(idx, 2)
    V = [A[i] for i in idx]
    return sparse(I, J, V, n, m)
end

""" Resize sparse vector b to (higher) dimension n. """
function resize_sparsevec(b, n)
    return sparsevec(b.nzind, b.nzval, n)
end

""" Approximative comparison of two matrices A and B. """
function isapprox(A::SparseMatrixCOO, B::SparseMatrixCOO)
    A2 = sparse(A)
    B2 = sparse(B, size(A2)...)
    return isapprox(A2, B2)
end

isapprox(A::SparseMatrixCOO, B) = isapprox(Matrix(A), B)
isapprox(A, B::SparseMatrixCOO) = isapprox(A, Matrix(B))
