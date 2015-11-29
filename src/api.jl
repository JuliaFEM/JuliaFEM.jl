# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

abstract JuliaFEMModel


type jMaterial
    name :: AbstractString
    scalar_data :: Dict{AbstractString, Float64}
    owners :: Vector
end

type jNode{T<:Real}
    id::Integer
    coords::Array{T, 1}
end

type jElement{T<:Real}
    id::Integer
    connectivity::Array{T, 1}
    element_type::AbstractString
end

type jNodeSet
    name::AbstractString
    node_ids::Array{Integer, 1} 
end

type jElementSet
    name :: AbstractString
    element_ids :: Array{Integer, 1}
end

type jProblem
    problem_type :: Symbol
    boundary_conditions :: Array{Dict{ASCIIString, Any}, 1} 
end


jProblem(a, b) = jProblem(symbol(a), b)

type jModel <: JuliaFEMModel
    nodes :: Vector{jNode}
    elements :: Vector{jElement}
    sets :: Dict{AbstractString, Union{jNodeSet, jElementSet}}
    material :: Dict{AbstractString, jMaterial}
    problems :: Array{jProblem}
    settings :: Dict{AbstractString, Real}
end

jModel() = jModel(jNode[],
                  jElement[],
                  Dict{AbstractString, Union{jNodeSet, jElementSet}}(),
                  Dict{AbstractString, jMaterial}(),
                  jProblem[],
                  Dict{AbstractString, Real}()
                  )

function build_core_model()

end

function build_core_elements()

end

function set_core_element_material()

end

function create_problems()

end

function add_problems!()

end
