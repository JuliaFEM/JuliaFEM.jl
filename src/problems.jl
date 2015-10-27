# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

abstract Problem
abstract BoundaryProblem <: Problem
abstract FieldProblem <: Problem

function get_equations(problem::Problem)
    problem.equations
end

function get_unknown_field_dimension(problem::Problem)
    problem.unknown_field_dimension
end

function get_unknown_field_name(problem::Problem)
    problem.unknown_field_name
end

""" Add new element to problem. """
function Base.push!(problem::Problem, element::Element)
    element_type = typeof(element)
    equation_type = problem.element_mapping[element_type]
    push!(problem.equations, equation_type(element))
end

