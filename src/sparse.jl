# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# Sparse utils to make assembly of local and global matrices easier.
# Unoptimized but should do all necessary stuff for at start.

type SparseMatrixCOO
    I :: Vector{Int}
    J :: Vector{Int}
    V :: Vector{Float64}
end

typealias SparseMatrixIJV SparseMatrixCOO

#=
function SparseMatrixIJV()
    warn("use SparseMatrixCOO to construct sparse matrix.""")
    SparseMatrixCOO([], [], [])
end
=#

function SparseMatrixCOO()
    SparseMatrixCOO([], [], [])
end

function Base.convert(::Type{SparseMatrixCOO}, A::SparseMatrixCSC)
    return SparseMatrixCOO(findnz(A)...)
end

function Base.sparse(A::SparseMatrixIJV, args...)
    return sparse(A.I, A.J, A.V, args...)
end

function Base.push!(A::SparseMatrixIJV, I::Int, J::Int, V::Float64)
    push!(A.I, I)
    push!(A.J, J)
    push!(A.V, V)
end

function Base.empty!(A::SparseMatrixIJV)
    empty!(A.I)
    empty!(A.J)
    empty!(A.V)
end

function Base.append!(A::SparseMatrixIJV, I::Vector{Int}, J::Vector{Int}, V::Vector{Float64})
    append!(A.I, I)
    append!(A.J, J)
    append!(A.V, V)
end

function Base.append!(A::SparseMatrixIJV, B::SparseMatrixIJV)
    append!(A.I, B.I)
    append!(A.J, B.J)
    append!(A.V, B.V)
end

function Base.isempty(A::SparseMatrixIJV)
    return isempty(A.I) && isempty(A.J) && isempty(A.V)
end

function Base.(:+)(A::SparseMatrixIJV, B::SparseMatrixIJV)
    if isempty(A)
        return B
    end
    if isempty(B)
        return A
    end
    C = SparseMatrixIJV([A.I;B.I], [A.J;B.J], [A.V;B.V])
    return C
end

function Base.full(A::SparseMatrixIJV, args...)
    return full(sparse(A.I, A.J, A.V, args...))
end

""" Add local element matrix to sparse matrix. This basically does:

>>> A[dofs1, dofs2] = A[dofs1, dofs2] + data

Example
-------

>>> S = [3, 4]
>>> M = [6, 7, 8]
>>> data = Float64[5 6 7; 8 9 10]
>>> A = SparseMatrixIJV()
>>> add!(A, S, M, data)
>>> full(A)
4x8 Array{Float64,2}:
 0.0  0.0  0.0  0.0  0.0  0.0  0.0   0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0   0.0
 0.0  0.0  0.0  0.0  0.0  5.0  6.0   7.0
 0.0  0.0  0.0  0.0  0.0  8.0  9.0  10.0

"""
function add!(A::SparseMatrixIJV, dofs1::Vector{Int}, dofs2::Vector{Int}, data::Matrix{Float64})
    n, m = size(data)
    for j=1:m
        for i=1:n
            push!(A.I, dofs1[i])
            push!(A.J, dofs2[j])
        end
    end
#   append!(A.I, repeat(dofs1, outer=[m]))
#   append!(A.J, repeat(dofs2, inner=[n]))
    append!(A.V, vec(data))
end

""" Add new data to COO Sparse vector. """
function add!(A::SparseMatrixCOO, dofs::Vector{Int}, data::Array{Float64})
    if length(dofs) != length(data)
        info("dofs = $dofs")
        info("data = $(vec(data))")
        error("when adding to sparse vector dimension mismatch!")
    end
    append!(A.I, dofs)
    append!(A.J, ones(Int, length(dofs)))
    append!(A.V, vec(data))
end

function optimize!(A::SparseMatrixIJV)
#   dim1 = length(A.I)
    I, J, V = findnz(sparse(A))
#   dim2 = length(I)
    A = SparseMatrixCOO(I, J, V)
    gc()
end
