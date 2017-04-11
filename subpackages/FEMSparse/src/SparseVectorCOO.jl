# This file is a part of JuliaFEM/FEMAssemble
# License is MIT: see https://github.com/JuliaFEM/FEMAssemble.jl/blob/master/LICENSE.md

import Base: size, resize!, convert, getindex, get!
import Base.SparseArrays: nonzeroinds, nonzeros, nnz

type SparseVectorCOO{Tv,Ti<:Integer} <: AbstractSparseArray{Tv,Ti<:Integer,1}
    I :: Vector{Ti}
    V :: Vector{Tv}
    cnt :: Int64
    max_cnt :: Int64
end

function SparseVectorCOO()
    return SparseVectorCOO(Int64[], Float64[], 0, 0)
end

function SparseVectorCOO{Tv,Ti<:Integer}(I::Vector{Ti}, V::Vector{Tv})
    @assert length(I) == length(V)
    return SparseVectorCOO(I, V, length(V), length(V))
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

""" Resize I,V vectors of SparseVectorCOO. """
function resize!(b::SparseVectorCOO, N::Int)
    if N < b.max_cnt
        warn("Down-sizing SparseVectorCOO")
    end
    b.max_cnt = N
    Base.resize!(b.I, b.max_cnt)
    Base.resize!(b.V, b.max_cnt)
    return b
end

""" Make sure that sparse matrix has space for at least N new enties. """
function reserve_space!(b::SparseVectorCOO, N::Int)
    resize!(b, b.cnt + N)
end

""" Add new value to sparse matrix. """
function add!{T}(b::SparseVectorCOO{T}, I::Int, V::T; auto_resize=true)
    if b.cnt == b.max_cnt
        if auto_resize
            resize!(b, b.cnt+1)
        else
            error("Cannot add to SparseVectorCOO: maximum size of $(b.cnt) exceeded.")
        end
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
        add!(A, dofs[i], data[i])
    end
end

""" Add SparseVector to SparseVectorCOO. """
function add!(a::SparseVectorCOO, b::SparseVector)
    I, V = findnz(b)
    c = SparseVectorCOO(I, V)
    append!(a, c)
    return
end
