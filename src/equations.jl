# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

abstract Equation

"""
Integration point

xi :: Array{Float64, 1}
    (dimensionless) coordinates of integration point
weight :: Float64
    Integration weight
attributes :: Dict{Any, Any}
    This is used to save internal variables of IP needed e.g. for incremental
    material models.
"""
type IntegrationPoint
    xi :: Array{Float64, 1}
    weight :: Float64
    attributes :: Dict{Any, Any}
end
IntegrationPoint(xi, weight) = IntegrationPoint(xi, weight, Dict{ASCIIString, Any}())

has_lhs(eq::Equation) = false
get_lhs(eq::Equation, xi) = nothing
has_rhs(eq::Equation) = false
get_rhs(eq::Equation, xi) = nothing
get_element(eq::Equation) = eq.element
get_integration_points(eq::Equation) = eq.integration_points
# couple convenient functions -- could make weak form definition easier
get_basis(eq::Equation, ip::IntegrationPoint) = get_basis(get_element(eq), ip.xi)
get_dbasisdx(eq::Equation, ip::IntegrationPoint) = get_dbasisdx(get_element(eq), ip.xi)
interpolate(eq::Equation, field::Union(ASCIIString, Symbol), ip::IntegrationPoint) = interpolate(get_element(el), field, ip.xi)
integrate_lhs(eq::Equation) = has_lhs(eq) ? integrate(eq, get_lhs) : nothing
integrate_rhs(eq::Equation) = has_rhs(eq) ? integrate(eq, get_rhs) : nothing

"""
Return determinant of Jacobian for numerical integration.
"""
function get_detJ(eq::Equation, ip::IntegrationPoint)
    el = get_element(eq)
    get_detJ(el, ip)
end
function get_detJ(el::Element, ip::IntegrationPoint)
    J = get_jacobian(el, ip.xi)
    n, m = size(J)
    if n != m # for manifolds
        return norm(J)
    else
        return det(J)
    end
end

"""
Integrate f over element

Parameters
----------
eq::Equation
    
f::Function
    Function to integrate
"""
function integrate(eq::Equation, f::Function)
    target = []
    for ip in get_integration_points(eq)
        push!(target, ip.weight*f(eq, ip)*get_detJ(eq, ip))
    end
    return sum(target)
end

