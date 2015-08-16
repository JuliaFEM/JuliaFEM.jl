# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md


type Element
    id :: Int
#   element_type :: Int
    node_ids :: Array{Int, 1}
    basis :: Function
    dbasis :: Function
    attributes :: Dict{ASCIIString, Any}
    ipoints :: Array{Float64, 2}
    iweights :: Array{Float64, 1}
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

