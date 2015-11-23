# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# Dirichlet boundary conditions in weak form

abstract DirichletEquation <: BoundaryEquation

### Dirichlet problem + equations

type DirichletProblem <: BoundaryProblem
    unknown_field_name :: ASCIIString
    unknown_field_dimension :: Int
    equations :: Vector{DirichletEquation}
end

""" Initialize new Dirichlet boundary condition.

Parameters
----------
dimension
    dimension of unknown field (scalar, vector, ...)

Examples
--------

"""
function DirichletProblem(unknown_field_name::ASCIIString, dimension::Int=1)
    DirichletProblem(unknown_field_name, dimension, [])
end

""" Dirichlet boundary condition element for 2 node line segment """
type DBC2D2 <: DirichletEquation
    element :: Seg2
    integration_points :: Vector{IntegrationPoint}
end

function Base.size(equation::DBC2D2)
    return (1, 2)
end

function Base.convert(::Type{DirichletEquation}, element::Seg2)
    integration_points = line3()
    haskey(element, "reaction force") || (element["reaction force"] = 0.0 => zeros(2))
    DBC2D2(element, integration_points)
end


function assemble!(assembly::Assembly, equation::DirichletEquation, time::Number=0.0, problem=nothing)
    isa(problem, Void) && error("Dicihlet boundary condition needs problem defined")
    field_dim = problem.unknown_field_dimension
    field_name = problem.unknown_field_name
    element = get_element(equation)
    gdofs = get_gdofs(element, field_dim)
    basis = get_basis(element)
    detJ = det(basis)
    for ip in get_integration_points(equation)
        w = ip.weight * detJ(ip)
        N = basis(ip, time)
        A = w*N'*N

        if haskey(element, field_name)
            # add all dimensions at once
            for i=1:field_dim
                g = element(field_name, ip, time)
                ldofs = gdofs[i:field_dim:end]
                add!(assembly.stiffness_matrix, ldofs, ldofs, A)
                add!(assembly.force_vector, ldofs, w*g*N')
            end
        end
  
        for i=1:field_dim
            if haskey(element, field_name*" $i")
                # add single component
                g = element(field_name*" $i", ip, time)
                ldofs = gdofs[i:field_dim:end]
                add!(assembly.stiffness_matrix, ldofs, ldofs, A)
                add!(assembly.force_vector, ldofs, w*g*N')
            end
        end
    end
end

