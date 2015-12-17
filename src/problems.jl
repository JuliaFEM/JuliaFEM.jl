# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

abstract AbstractProblem

type Problem{T<:AbstractProblem}
    name :: ASCIIString
    dim :: Int
    elements :: Vector{Element}
end

type BoundaryProblem{T<:AbstractProblem}
    name :: ASCIIString
    parent_field_name :: ASCIIString
    parent_field_dim :: Int
    dim :: Int
    elements :: Vector{Element}
end

typealias FieldProblem Problem

typealias AllProblems Union{Problem, BoundaryProblem}

function get_elements(problem::AllProblems)
    return problem.elements
end

""" Return the dimension of the unknown field of this problem. """
function get_unknown_field_dimension(problem::Problem)
    return problem.dim
end

""" Return the name of the unknown field of this problem. """
function get_unknown_field_name{P<:AbstractProblem}(problem::Problem{P})
    return get_unknown_field_name(P)
end

function Base.push!(problem::AllProblems, element::Element)
    push!(problem.elements, element)
end

