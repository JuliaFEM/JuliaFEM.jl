# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# Heat problems

abstract HeatProblem <: Problem
abstract HeatEquation <: Equation

### Plane heat problem + equations ###

type PlaneHeatProblem <: HeatProblem
    unknown_field_name :: ASCIIString
    unknown_field_dimension :: Int
    equations :: Array{HeatEquation, 1}
    element_mapping :: Dict{DataType, DataType}
end

""" Default constructor for problem takes no arguments. """
function PlaneHeatProblem()
    element_mapping = Dict(
        Quad4 => DC2D4,
        Seg2 => DC2D2)
    return PlaneHeatProblem("temperature", 1, [], element_mapping)
end



""" Diffusive heat transfer for 4-node bilinear element. """
type DC2D4 <: HeatEquation
    element :: Quad4
    integration_points :: Array{IntegrationPoint, 1}
end
function DC2D4(element::Quad4)
    integration_points = get_default_integration_points(element)
    push!(element, FieldSet("temperature"))
    DC2D4(element, integration_points)
end
Base.size(equation::DC2D4) = (1, 4)

""" Diffusive heat transfer for 2-node linear segment. """
type DC2D2 <: HeatEquation
    element :: Seg2
    integration_points :: Array{IntegrationPoint, 1}
end
function DC2D2(element::Seg2)
    integration_points = get_default_integration_points(element)
    push!(element, FieldSet("temperature"))
    DC2D2(element, integration_points)
end
Base.size(equation::DC2D2) = (1, 2)

function calculate_local_assembly!(assembly::LocalAssembly, equation::HeatEquation,
                                            unknown_field_name::ASCIIString, time::Number=Inf,
                                            problem=nothing)

    initialize_local_assembly!(assembly, equation)

    element = get_element(equation)
    basis = get_basis(element)
    dbasis = grad(basis)
    detJ = det(basis)
    for ip in get_integration_points(equation)
        w = ip.weight * detJ(ip)
        # evaluate fields in integration point
        ρ = basis("density", ip, time)
        k = basis("temperature thermal conductivity", ip, time)
        f = basis("temperature load", ip, time)
        # evaluate basis functions and gradient in integration point
        N = basis(ip, time)
        dN = dbasis(ip, time)
        # do assembly
        assembly.mass_matrix += w * ρ*N'*N
        assembly.stiffness_matrix += w * k*dN'*dN
        assembly.force_vector += w * N'*f
    end
end

