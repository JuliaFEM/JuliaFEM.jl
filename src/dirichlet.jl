# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# Dirichlet boundary conditions in weak form

abstract DirichletEquation <: Equation

function get_unknown_field_name(equation::DirichletEquation)
    return "reaction force"
end

### Dirichlet problem + equations

type DirichletProblem <: BoundaryProblem
    unknown_field_name :: ASCIIString
    unknown_field_dimension :: Int
    equations :: Vector{DirichletEquation}
#   element_mapping :: Dict{DataType, DataType}
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
function DirichletProblem(dimension::Int=1, field_value::Function=(X)->[0.0,0.0,0.0], equations=[])
#    element_mapping = nothing
#    if dimension == 1
#        element_mapping = Dict(
#            Seg2 => DBC2D2
#            )
#    end
    DirichletProblem("reaction force", dimension, equations, field_value)
#    DirichletProblem("reaction force", dimension, [], element_mapping, field_value)
end

""" Dirichlet boundary condition element for 2 node line segment """
type DBC2D2 <: DirichletEquation
    element :: Seg2
    integration_points :: Vector{IntegrationPoint}
end

function Base.size(equation::DBC2D2)
    return (1, 2)
end

#function DBC2D2(element::Seg2)
function Base.convert(::Type{DirichletEquation}, element::Seg2)
    integration_points = line3()
    haskey(element, "reaction force") || (element["reaction force"] = zeros(1, 2))
    DBC2D2(element, integration_points)
end


function assemble!(assembly::Assembly, equation::DirichletEquation, time::Number=0.0, problem=nothing)
    gdofs = get_gdofs(equation)
#   info("gdofs = $gdofs")
    element = get_element(equation)
    basis = get_basis(element)
    detJ = det(basis)
    for ip in get_integration_points(equation)
        w = ip.weight * detJ(ip)
        N = basis(ip, time)
        add!(assembly.stiffness_matrix, gdofs, gdofs, w*N'*N)
#       info("added $(w*N'*N)")
#        if !isa(problem, Void)
#            X = basis("geometry", ip, time)
#            u = problem.field_value(X)[1:length(gdofs)]
#            add!(assembly.force_vector, gdofs, w*N'*u)
#        end
    end
#   info("assembly done")
end

