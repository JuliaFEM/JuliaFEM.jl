# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

abstract calculation_model

type Model <: calculation_model
    jotain
end


type INode{T<:Real}
    id::Integer
    coords::Array{T, 1}
end

type IElement{T<:Real}
    id::Integer
    connectivity::Array{T, 1}
    element_type::AbstractString
    # fields :: Dict{AbstractString, Field}
end

# IElement(id, conn, eltype) = IElement(id, conn, eltype, Dict())

type INodeSet
    name::AbstractString
    node_ids::Array{Integer, 1} 
end

type IElementSet
    name::AbstractString
    element_ids::Array{Integer, 1}
end

