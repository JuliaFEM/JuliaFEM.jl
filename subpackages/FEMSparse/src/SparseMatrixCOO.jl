# This file is a part of JuliaFEM/SparseCOO
# License is MIT: see https://github.com/JuliaFEM/SparseCOO.jl/blob/master/LICENSE.md

import Base: convert, size, full, resize!, empty!, isempty, isapprox, findnz, getindex, sparse

global const SPARSEMATRIXCOO_DEFAULT_BLOCK_SIZE = 1024*1024

type SparseMatrixCOO{Tv,Ti<:Integer} <: AbstractSparseArray{Tv,Ti<:Integer,2}
    I :: Vector{Ti}
    J :: Vector{Ti}
    V :: Vector{Tv}
    cnt :: Int64
    max_cnt :: Int64
    blk_size :: Int64
end

function SparseMatrixCOO()
    return SparseMatrixCOO(Int64[], Int64[], Float64[], 0, 0, SPARSEMATRIXCOO_DEFAULT_BLOCK_SIZE)
end

function SparseMatrixCOO{Tv,Ti<:Integer}(I::Vector{Ti}, J::Vector{Ti}, V::Vector{Tv})
    @assert length(I) == length(J) == length(V)
    return SparseMatrixCOO(I, J, V, length(V), length(V), SPARSEMATRIXCOO_DEFAULT_BLOCK_SIZE)
end

function SparseMatrixCOO{Tv,Ti<:Integer}(A::SparseMatrixCSC{Tv,Ti})
    I, J, V = findnz(A)
    return SparseMatrixCOO(I, J, V)
end

function convert{Tv,Ti<:Integer}(::Type{SparseMatrixCOO{Tv,Ti}}, A::SparseMatrixCSC{Tv,Ti})
    I, J, V = findnz(A)
    return SparseMatrixCOO(I, J, V)
end

function SparseMatrixCOO{T}(A::Matrix{T})
    return SparseMatrixCOO(sparse(A))
end

function getindex(A::SparseMatrixCOO, I::Int64, J::Int64)
    error("Indexing of SparseMatrixCOO is very ineffective and not implemented. Convert this to SparseMatrixCSC.")
end

function findnz(A::SparseMatrixCOO)
    return (A.I[1:A.cnt], A.J[1:A.cnt], A.V[1:A.cnt])
end

function size(A::SparseMatrixCOO)
    isempty(A) && return (0, 0)
    return maximum(A.I), maximum(A.J)
end

function size(A::SparseMatrixCOO, idx::Int)
    return size(A)[idx]
end

function full{Tv,Ti<:Integer}(A::SparseMatrixCOO{Tv,Ti}, args...)
    if A.cnt == 0
        return Matrix{Tv}()
    end
    return full(sparse(A, args...))
end

""" Return Julia default SparseMatrixCSC from SparseMatrixCOO. """
function sparse(A::SparseMatrixCOO, args...)
    return sparse(A.I[1:A.cnt], A.J[1:A.cnt], A.V[1:A.cnt], args...)
end

""" Allocate (more) memory for I,J,V vectors of SparseMatrixCOO. """
function allocate!(A::SparseMatrixCOO, N::Int)
    if N < A.max_cnt
        warn("Down-sizing SparseMatrixCOO")
    end
    A.max_cnt = N
    resize!(A.I, A.max_cnt)
    resize!(A.J, A.max_cnt)
    resize!(A.V, A.max_cnt)
    return A
end

""" Add new value to sparse matrix. """
function add!{T}(A::SparseMatrixCOO{T}, I::Int, J::Int, V::T)
    if A.cnt == A.max_cnt
        allocate!(A, A.cnt+A.blk_size)
    end
    A.cnt += 1
    @inbounds begin
        A.I[A.cnt] = I
        A.J[A.cnt] = J
        A.V[A.cnt] = V
    end
    return A
end

function empty!(A::SparseMatrixCOO)
    empty!(A.I)
    empty!(A.J)
    empty!(A.V)
    A.cnt = 0
    A.max_cnt = 0
    return A
end

""" Add one or more SparseMatrixCOOs to one SparseMatrixCOO. """
function add!(A::SparseMatrixCOO, rest::SparseMatrixCOO...)
    for B in rest
        for (I, J, V) in zip(B.I[1:B.cnt], B.J[1:B.cnt], B.V[1:B.cnt])
            add!(A, I, J, V)
        end
    end
end

function isempty(A::SparseMatrixCOO)
    return A.cnt == 0
end

""" Add local element matrix to sparse matrix. This basically does:

>>> A[dofs1, dofs2] = A[dofs1, dofs2] + data

Example
-------

>>> S = [3, 4]
>>> M = [6, 7, 8]
>>> data = Float64[5 6 7; 8 9 10]
>>> A = SparseMatrixCOO()
>>> add!(A, S, M, data)
>>> full(A)
4x8 Array{Float64,2}:
 0.0  0.0  0.0  0.0  0.0  0.0  0.0   0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0   0.0
 0.0  0.0  0.0  0.0  0.0  5.0  6.0   7.0
 0.0  0.0  0.0  0.0  0.0  8.0  9.0  10.0

"""
function add!(A::SparseMatrixCOO, dofs1::Vector{Int}, dofs2::Vector{Int}, data::Matrix)
    n, m = size(data)
    @assert length(dofs1) == n
    @assert length(dofs2) == m
    for j=1:m
        for i=1:n
            @inbounds add!(A, dofs1[i], dofs2[j], data[i,j])
        end
    end
    return nothing
end

""" Add sparse matrix of CSC to COO. """
function add!(A::SparseMatrixCOO, B::SparseMatrixCSC)
    I, J, V = findnz(B)
    add!(A, I, J, V)
end

function get_nonzero_rows(A::SparseMatrixCOO)
    return unique(A.I[1:A.cnt])
end

function get_nonzero_columns(A::SparseMatrixCOO)
    return unique(A.J[1:A.cnt])
end
