# This file is a part of JuliaFEM/FEMAssemble
# License is MIT: see https://github.com/JuliaFEM/FEMAssemble.jl/blob/master/LICENSE.md

type SparseVectorDOK{Tv,Ti<:Integer} <: AbstractSparseArray{Tv,Ti<:Integer,1}
    data :: Dict{Ti,Tv}
end

function SparseVectorDOK()
    return SparseVectorDOK(Dict{Int64, Float64}())
end

function SparseVectorDOK{Tv,Ti<:Integer}(b::SparseVector{Tv,Ti})
    I, V = findnz(b)
    c = SparseVectorDOK()
    add!(c, I, V)
    return c
end

function SparseVectorDOK{T}(b::Vector{T})
    return SparseVectorDOK(sparsevec(b))
end

function add!{Tv,Ti<:Integer}(b::SparseVectorDOK{Tv,Ti}, i::Ti, v::Tv)
    b.data[i] = get(b, i) + v
    return nothing
end

function add!{Tv,Ti<:Integer}(b::SparseVectorDOK{Tv,Ti}, dofs::Vector{Ti}, data::Vector{Tv})
    @assert length(dofs) == length(data)
    z = Tv(0)
    for i=1:length(dofs)
        @inbounds b.data[dofs[i]] = Base.get(b.data, i, z) + data[i]
    end
    return nothing
end

function get{Tv,Ti<:Integer}(b::SparseVectorDOK{Tv,Ti}, i::Ti)
    return Base.get(b.data, i, Tv(0))
end

function get!{Tv,Ti<:Integer}(b::SparseVectorDOK{Tv,Ti}, dofs::Vector{Ti}, data::Vector{Tv})
    for (i,j) in enumerate(dofs)
        data[i] = get(b, i)
    end
    return nothing
end

