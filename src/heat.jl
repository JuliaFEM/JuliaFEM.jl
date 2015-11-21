# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# Heat problems

abstract HeatProblem <: Problem
abstract HeatEquation <: Equation

function get_unknown_field_name(equation::HeatEquation)
    return "temperature"
end

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
function assemble!(assembly::Assembly, equation::HeatEquation, time::Number=0.0, problem=nothing)

    element = get_element(equation)
    gdofs = get_gdofs(equation)
    basis = get_basis(element)
    dbasis = grad(basis)
    detJ = det(basis)
    for ip in get_integration_points(equation)
        w = ip.weight*detJ(ip)
        N = basis(ip, time)
        if haskey(element, "density")
            rho = basis("density", ip, time)
            add!(assembly.mass_matrix, gdofs, gdofs, w*rho*N'*N)
        end
        if haskey(element, "temperature thermal conductivity")
            dN = dbasis(ip, time)
            k = basis("temperature thermal conductivity", ip, time)
            add!(assembly.stiffness_matrix, gdofs, gdofs, w*k*dN'*dN)
        end
        if haskey(element, "temperature load")
            f = basis("temperature load", ip, time)
            add!(assembly.force_vector, gdofs, w*N'*f)
        end
        if haskey(element, "temperature flux")
            info("assemble boundary flux")
            g = basis("temperature flux", ip, time)
            add!(assembly.force_vector, gdofs, w*N'*g)
        end
    end
end


### Equations ###

""" Diffusive heat transfer for 4-node bilinear element. """
type DC2D4 <: HeatEquation
    element :: Quad4
    integration_points :: Vector{IntegrationPoint}
end

function Base.size(equation::DC2D4)
    return (1, 4)
end

""" Diffusive heat transfer for 2-node linear segment. """
type DC2D2 <: HeatEquation
    element :: Seg2
    integration_points :: Vector{IntegrationPoint}
end

function Base.size(equation::DC2D2)
    return (1, 2)
end

# Conversions element -> equation

function Base.convert(::Type{HeatEquation}, element::Quad4)
    integration_points = get_default_integration_points(element)
    haskey(element, "temperature") || (element["temperature"] = 0.0 => zeros(4))
    DC2D4(element, integration_points)
end

function Base.convert(::Type{HeatEquation}, element::Seg2)
    integration_points = get_default_integration_points(element)
    haskey(element, "temperature") || (element["temperature"] = 0.0 => zeros(2))
    DC2D2(element, integration_points)
end

### Problems ###

type PlaneHeatProblem <: HeatProblem
    unknown_field_name :: ASCIIString
    unknown_field_dimension :: Int
    equations :: Vector{HeatEquation}
end

""" Default constructor for problem takes no arguments. """
function PlaneHeatProblem(equations=[])
    return PlaneHeatProblem("temperature", 1, equations)
end
