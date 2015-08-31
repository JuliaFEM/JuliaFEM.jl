# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

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

