# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# Dirichlet boundary conditions in weak form

abstract DirichletEquation <: Equation

### Dirichlet problem + equations

type DirichletProblem <: BoundaryProblem
    unknown_field_name :: ASCIIString
    unknown_field_dimension :: Int
    equations :: Vector{DirichletEquation}
    element_mapping :: Dict{Element, Equation}
    field_value :: Function
end

""" Initialize new Dirichlet boundary condition.

Parameters
----------
dimension
    dimension of unknown field
field_value
    boundary function

Examples
--------

Create u(X) = 0.0 boundary condition for three-dimensional elasticity problem:

>>> u(X) = [0.0, 0.0, 0.0]
>>> bc = DirichletProblem(3, u)
"""
function DirichletProblem(dimension::Int=1, field_value::Function=(X)->[0.0,0.0,0.0])
    element_mapping = nothing
    if dimension == 1
        element_mapping = Dict(
            Seg2 => DBC2D2
            )
    end
    DirichletProblem("reaction force", dimension, [], element_mapping, field_value)
end

""" Dirichlet boundary condition element for 2 node line segment """
type DBC2D2 <: DirichletEquation
    element :: Seg2
    integration_points :: Vector{IntegrationPoint}
end
function DBC2D2(element::Seg2)
    integration_points = default_integration_points(element)
    if !haskey(element, "reaction force")
        element["reaction force"] = zeros(1, 2)
    end
    DBC2D2(element, integration_points)
end
Base.size(equation::DBC2D2) = (1, 2)

function calculate_local_assembly!(assembly::LocalAssembly, equation::DirichletEquation, unknown_field_name::ASCIIString, time::Number=0.0, problem=nothing)
    initialize_local_assembly!(assembly, equation)
    element = get_element(equation)
    basis = get_basis(element)
    detJ = det(basis)
    for ip in get_integration_points(equation)
        w = ip.weight * detJ(ip)
        N = basis(ip, time)
        assembly.stiffness_matrix += w * N'*N
        if !isa(problem, Void)
            X = basis("geometry", ip, time)
            u = problem.field_value(X)
            assembly.force_vector += w * N'*u
        end
    end
end

