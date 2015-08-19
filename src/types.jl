# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

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

type Element
    id :: Int
#   element_type :: Int
    node_ids :: Array{Int, 1}
    basis :: Function
    dbasis :: Function
    integration_points :: Array{IntegrationPoint, 1}
    attributes :: Dict{ASCIIString, Any}
#   ipoints :: Array{Float64, 2}
#   iweights :: Array{Float64, 1}
end


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

