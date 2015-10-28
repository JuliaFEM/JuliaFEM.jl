# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

abstract Problem
abstract BoundaryProblem <: Problem
abstract FieldProblem <: Problem

""" Return all equations beloging to this problem. """
function get_equations(problem::Problem)
    problem.equations
end

""" Return the dimension of the unknown field of this problem. """
function get_unknown_field_dimension(problem::Problem)
    problem.unknown_field_dimension
end

""" Return the name of the unknown field of this problem. """
function get_unknown_field_name(problem::Problem)
    problem.unknown_field_name
end

"""
Add new equation to problem.

Parameters
----------
problem
element

Notes
-----
Equation is automatically created during process based on problem
element -> equation mapping and element type.
"""
function Base.push!(problem::Problem, element::Element)
    element_type = typeof(element)
    equation_type = problem.element_mapping[element_type]
    push!(problem.equations, equation_type(element))
end

"""
Return the size of the problem, i.e.
maximum number of connectivity × unknown field dimension.
"""
function Base.size(problem::Problem)
    mc = 0
    for equation in get_equations(problem)
        element = get_element(equation)
        mc = max(mc, get_connectivity(element)...)
    end
    dim = get_unknown_field_dimension(problem)
    return (dim, dim*mc)
end

""" Assign new equation mapping to problem for some element.

Examples
--------
>>> p = PlaneHeatProblem()
>>> p[Seg2] = DC2D2

"""
function Base.setindex!(problem::Problem, equation, element)
    problem.element_mapping[element] = equation
end

""" Return global degrees of freedom of element in matrix level. """
function get_gdofs(problem::Problem, equation::Equation)
    dim = get_unknown_field_dimension(problem)
    element = get_element(equation)
    conn = get_connectivity(element)
    gdofs = vec(vcat([dim*conn'-i for i=dim-1:-1:0]...))
    return gdofs
end

