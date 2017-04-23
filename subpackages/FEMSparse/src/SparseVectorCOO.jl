# This file is a part of JuliaFEM/FEMAssemble
# License is MIT: see https://github.com/JuliaFEM/FEMAssemble.jl/blob/master/LICENSE.md

import Base: size, resize!, convert, getindex, get!, isempty, empty!, sparsevec, full
import Base.SparseArrays: nonzeroinds, nonzeros, nnz

global const SPARSEVECTORCOO_DEFAULT_BLOCK_SIZE = 1024

type SparseVectorCOO{Tv,Ti<:Integer} <: AbstractSparseArray{Tv,Ti<:Integer,1}
    I :: Vector{Ti}
    V :: Vector{Tv}
    cnt :: Int64
    max_cnt :: Int64
    blk_size :: Int64
end

function SparseVectorCOO()
    return SparseVectorCOO(Int64[], Float64[], 0, 0, SPARSEVECTORCOO_DEFAULT_BLOCK_SIZE)
end

function SparseVectorCOO{Tv,Ti<:Integer}(I::Vector{Ti}, V::Vector{Tv})
    @assert length(I) == length(V)
    return SparseVectorCOO(I, V, length(V), length(V), SPARSEVECTORCOO_DEFAULT_BLOCK_SIZE)
end

function SparseVectorCOO{T}(b::Vector{T})
    return SparseVectorCOO(sparsevec(b))
end

function SparseVectorCOO{Tv,Ti<:Integer}(b::SparseVector{Tv,Ti})
    I, V = findnz(b)
    return SparseVectorCOO(I, V)
end

function convert{Tv,Ti<:Integer}(::Type{SparseVectorCOO{Tv,Ti}}, b::SparseVector{Tv,Ti})
    I, V = findnz(b)
    return SparseVectorCOO(I, V)
end

function size(b::SparseVectorCOO)
    if b.cnt == 0
        return (0,)
    end
    return (maximum(b.I),)
end

function nnz(b::SparseVectorCOO)
    return length(nonzeroinds(b))
end

function nonzeroinds(b::SparseVectorCOO)
    return unique(b.I)
end

function nonzeros(b::SparseVectorCOO)
    return nonzeros(sparsevec(b.I, b.V))
end

function isempty(b::SparseVectorCOO)
    return b.cnt == 0
end

function full(b::SparseVectorCOO, args...)
    full(sparsevec(b, args...))
end

function empty!(b::SparseVectorCOO)
    empty!(b.I)
    empty!(b.V)
    b.cnt = 0
    b.max_cnt = 0
    return b
end

function sparsevec(b::SparseVectorCOO, args...)
    return Base.sparsevec(b.I[1:b.cnt], b.V[1:b.cnt], args...)
end

#=

function get!{Tv,Ti<:Integer}(b::SparseVectorCOO{Tv,Ti}, dofs::Vector{Ti}, data::Vector{Tv})
    @assert length(dofs) == length(data)
    for i=1:length(data)
        data[i] = get(b, i)
    end
    return nothing
end

function get{Tv,Ti<:Integer}(b::SparseVectorCOO{Tv,Ti}, idx::Ti)
    value = 0.0
    for i=1:b.cnt
        if b.I[i] == idx
            value += b.V[i]
        end
    end
    return value
end

=#

""" Allocate memory for I,V vectors of SparseVectorCOO. """
function allocate!(b::SparseVectorCOO, N::Int)
    if N < b.max_cnt
        warn("Down-sizing SparseVectorCOO vectors")
    end
    b.max_cnt = N
    Base.resize!(b.I, b.max_cnt)
    Base.resize!(b.V, b.max_cnt)
    return b
end

""" Add new value to sparse matrix. """
function add!{T}(b::SparseVectorCOO{T}, I::Int, V::T)
    if b.cnt == b.max_cnt
        allocate!(b, b.cnt+b.blk_size)
    end
    b.cnt += 1
    @inbounds begin
        b.I[b.cnt] = I
        b.V[b.cnt] = V
    end
    return b
end

""" Add new data to COO Sparse vector. """
function add!(A::SparseVectorCOO, dofs::Vector{Int}, data::Array{Float64})
    @assert length(dofs) == length(data)
    for i=1:length(data)
        @inbounds add!(A, dofs[i], data[i])
    end
    return nothing
end

""" Add SparseVector to SparseVectorCOO. """
function add!(a::SparseVectorCOO, b::SparseVector)
    I, V = findnz(b)
    for (i, v) in zip(I, V)
        add!(a, i, v)
    end
    return nothing
end

""" Add one or more SparseVectorCOO to SparseVectorCOO. """
function add!(a::SparseVectorCOO, rest::SparseVectorCOO...)
    for b in rest
        for (i, v) in zip(b.I[1:b.cnt], b.V[1:b.cnt])
            add!(a, i, v)
        end
    end
    return nothing
end
