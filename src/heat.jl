# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# Heat problems

abstract HeatProblem <: Problem
abstract HeatEquation <: Equation

### Formulation ###

""" Heat equations.

Formulation
-----------

Field equation is:

    ρc∂u/∂t = ∇⋅(k∇u) + f

Weak form is: find u∈U such that ∀v in V

    ∫k∇u∇v dx = ∫fv dx + ∫gv ds,

where

    k = temperature thermal conductivity    defined on volume
    f = temperature load                    defined on volume
    g = temperature flux                    defined on boundary

References
----------
https://en.wikipedia.org/wiki/Heat_equation

"""
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
        N = basis(ip, time)
        if haskey(element, "density")
            ρ = basis("density", ip, time)
            assembly.mass_matrix += w * ρ*N'*N
        end
        if haskey(element, "temperature thermal conductivity")
            dN = dbasis(ip, time)
            k = basis("temperature thermal conductivity", ip, time)
            assembly.stiffness_matrix += w * k*dN'*dN
        end
        if haskey(element, "temperature load")
            f = basis("temperature load", ip, time)
            assembly.force_vector += w * N'*f
        end
        if haskey(element, "temperature flux")
            g = basis("temperature flux", ip, time)
            assembly.force_vector += w * N'*g
        end
    end
end

### Problems ###

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

### Equations ###

""" Diffusive heat transfer for 4-node bilinear element. """
type DC2D4 <: HeatEquation
    element :: Quad4
    integration_points :: Array{IntegrationPoint, 1}
end
function DC2D4(element::Quad4)
    integration_points = get_default_integration_points(element)
    if !haskey(element, "temperature")
        element["temperature"] = FieldSet()
    end
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
    if !haskey(element, "temperature")
        element["temperature"] = FieldSet()
    end
    DC2D2(element, integration_points)
end
Base.size(equation::DC2D2) = (1, 2)

