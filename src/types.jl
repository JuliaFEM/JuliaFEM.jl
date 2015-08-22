# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

export IntegrationPoint, Element, Assembly, FunctionSpace

"""
Integration point

xi :: Array{Float64, 1}
    (dimensionless) coordinates of integration point
weight :: Float64
    Integration weight
attributes :: Dict{ASCIIString, Any}
    This is used to save internal variables of IP needed e.g. for incremental
    material models.
"""
type IntegrationPoint
    xi :: Array{Float64, 1}
    weight :: Float64
    attributes :: Dict{ASCIIString, Any}
end

type FunctionSpace
    basis :: Function
    dbasis :: Function
end

abstract Element

#type Element
#    id :: Int
#    node_ids :: Array{Int, 1}
#    shape_functions :: FunctionSpace
#    integration_points :: Array{IntegrationPoint, 1}
#    attributes :: Dict{ASCIIString, Any}
#end


type Assembly
    # LHS
    I :: Array{Int64, 1}
    J :: Array{Int64, 1}
    A :: Array{Float64, 1}
    # RHS
    i :: Array{Int64, 1}
    b :: Array{Float64, 1}
    # global dofs for each element
    gdofs :: Dict{Int64, Array{Int64, 1}}
end
Assembly() = Assembly(Int64[], Int64[], Float64[], Int64[], Float64[], Dict{Int64,Array{Int64,1}}())
Assembly(gdofs::Dict{Int64,Array{Int64,1}}) = Assembly(Int64[], Int64[], Float64[], Int64[], Float64[], gdofs)

