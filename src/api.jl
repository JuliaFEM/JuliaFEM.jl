# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md


type INode{T<:Real}
    id::Integer
    coords::Array{T, 1}
end

type IElement{T<:Real}
    id::Integer
    connectivity::Array{T, 1}
    element_type::AbstractString
end


type INodeset
    node_ids
end

type IElementset
    jotain
end

