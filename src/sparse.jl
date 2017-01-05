# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# Sparse utils to make assembly of local and global matrices easier.
# Unoptimized but should do all necessary stuff for at start.

type SparseMatrixCOO{T<:Real}
    I :: Vector{Int}
    J :: Vector{Int}
    V :: Vector{T}
end

typealias SparseVectorCOO SparseMatrixCOO

function SparseMatrixCOO()
    return SparseMatrixCOO{Float64}([], [], [])
end

function SparseVectorCOO(I::Vector, V::Vector)
    return SparseVectorCOO(I, ones(I), V)
end

function convert(::Type{SparseMatrixCOO}, A::SparseMatrixCSC)
    return SparseMatrixCOO(findnz(A)...)
end

function convert(::Type{SparseMatrixCOO}, A::Matrix)
    return SparseMatrixCOO(findnz(A)...)
end

function convert(::Type{SparseMatrixCOO}, A::Vector)
    return SparseMatrixCOO(findnz(sparse(A))...)
end

""" Convert from COO format to CSC.

Parameters
----------
tol
    used to drop near zero values less than tol.
"""
function sparse(A::SparseMatrixCOO, args...; tol=1.0e-12)
    B = sparse(A.I, A.J, A.V, args...)
    SparseArrays.droptol!(B, tol)
    return B
end

function push!(A::SparseMatrixCOO, I::Int, J::Int, V::Float64)
    push!(A.I, I)
    push!(A.J, J)
    push!(A.V, V)
end

function empty!(A::SparseMatrixCOO)
    empty!(A.I)
    empty!(A.J)
    empty!(A.V)
end

function append!(A::SparseMatrixCOO, I::Vector{Int}, J::Vector{Int}, V::Vector{Float64})
    append!(A.I, I)
    append!(A.J, J)
    append!(A.V, V)
end

function append!(A::SparseMatrixCOO, B::SparseMatrixCOO)
    append!(A.I, B.I)
    append!(A.J, B.J)
    append!(A.V, B.V)
end

function isempty(A::SparseMatrixCOO)
    return isempty(A.I) && isempty(A.J) && isempty(A.V)
end

function Base.:+(A::SparseMatrixCOO, B::SparseMatrixCOO)
    if isempty(A)
        return B
    end
    if isempty(B)
        return A
    end
    C = SparseMatrixCOO([A.I;B.I], [A.J;B.J], [A.V;B.V])
    return C
end

function full(A::SparseMatrixCOO, args...)
    return full(sparse(A.I, A.J, A.V, args...))
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
    for j=1:m
        for i=1:n
            push!(A.I, dofs1[i])
            push!(A.J, dofs2[j])
        end
    end
    append!(A.V, vec(data))
end

""" Add sparse matrix of CSC to COO. """
function add!(A::SparseMatrixCOO, B::SparseMatrixCSC)
    I, J, V = findnz(B)
    C = SparseMatrixCOO(I, J, V)
    append!(A, C)
end

""" Add new data to COO Sparse vector. """
function add!(A::SparseMatrixCOO, dofs::Vector{Int}, data::Array{Float64}, dim::Int=1)
    if length(dofs) != length(data)
        info("dofs = $dofs")
        info("data = $(vec(data))")
        error("when adding to sparse vector dimension mismatch!")
    end
    append!(A.I, dofs)
    append!(A.J, dim*ones(Int, length(dofs)))
    append!(A.V, vec(data))
end

""" Add SparseVector to SparseVectorCOO. """
function add!(a::SparseVectorCOO, b::SparseVector)
    I, V = findnz(b)
    c = SparseVectorCOO(I, V)
    append!(a, c)
    return
end

""" Combine (I,J,V) values if possible to reduce memory usage. """
function optimize!(A::SparseMatrixCOO)
    I, J, V = findnz(sparse(A))
    A.I = I
    A.J = J
    A.V = V
end

""" Find all nonzero rows from sparse matrix.

Returns
-------

Ordered list of row indices.
"""
function get_nonzero_rows(A::SparseMatrixCSC)
    return sort(unique(rowvals(A)))
end

function get_nonzero_columns(A::SparseMatrixCSC)
    return get_nonzero_rows(transpose(A))
end

function get_nonzero_rows(A::Union{SparseMatrixCOO, Matrix})
    return get_nonzero_rows(sparse(A))
end

function get_nonzero_columns(A::Union{SparseMatrixCOO, Matrix})
    return get_nonzero_columns(sparse(A))
end

function get_nonzeros(C::Union{SparseMatrixCSC, Matrix})
    nz1 = get_nonzero_rows(C)
    nz2 = get_nonzero_columns(C)
    return (nz1, nz2)
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
    return sparse(findnz(A)..., n, m)
end

""" Resize sparse vector b to (higher) dimension n. """
function resize_sparsevec(b, n)
    return sparsevec(findnz(b)..., n)
end

""" Matrix norm. Automatically convert to dense when asking for 2-norm for small matrices. """
function norm(A::SparseMatrixCOO, p=Inf; maxdim=1000)
    dim = size(A, 1)
    if p == 2 && dim > maxdim
        warn("Assembly norm: dim = $dim > $maxdim and p=$p, not making dense matrices for operation.")
        return 0.0
    end
    if p == 2
        return norm(full(A), p)
    else
        return norm(sparse(A), p)
    end
end

""" Approximative comparison of two matricse A and B. """
function isapprox(A::SparseMatrixCOO, B::SparseMatrixCOO)
    A2 = sparse(A)
    B2 = sparse(B, size(A2)...)
    return isapprox(A2, B2)
end
