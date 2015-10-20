# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

abstract Equation

function get_unknown_field_name(equation::Equation)
    eqtype = typeof(equation)
    error("define get_unknown_field_name for this equation type $eqtype")
end

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
get_connectivity(eq::Equation) = get_connectivity(get_element(eq))
get_basis(eq::Equation, ip::IntegrationPoint) = get_basis(get_element(eq), ip.xi)
get_dbasisdx(eq::Equation, ip::IntegrationPoint) = get_dbasisdx(get_element(eq), ip.xi)
interpolate(eq::Equation, field::Union{ASCIIString, Symbol}, ip::IntegrationPoint) = interpolate(get_element(el), field, ip.xi)
integrate_lhs(eq::Equation, t::Number) = has_lhs(eq) ? integrate(eq, get_lhs, t) : nothing
integrate_rhs(eq::Equation, t::Number) = has_rhs(eq) ? integrate(eq, get_rhs, t) : nothing
get_lhs(eq::Equation, t::Number) = has_lhs(eq) ? integrate(eq, get_lhs, t) : nothing
get_rhs(eq::Equation, t::Number) = has_rhs(eq) ? integrate(eq, get_rhs, t) : nothing


"""
Return determinant of Jacobian for numerical integration.
"""
function get_detJ(eq::Equation, ip::IntegrationPoint, t::Float64)
    el = get_element(eq)
    get_detJ(el, ip, t)
end
function get_detJ(el::Element, ip::IntegrationPoint, t::Float64)
    get_detJ(el, ip.xi, t)
end
function get_detJ(el::Element, xi::Vector, t::Float64)
    J = get_jacobian(el, xi, t)
    s = size(J)
    return s[1] == s[2] ? det(J) : norm(J)
end

"""
Integrate f over element

Parameters
----------
eq::Equation

f::Function
    Function to integrate
"""
function integrate(eq::Equation, f::Function, t::Float64)
    target = []
    for ip in get_integration_points(eq)
        push!(target, ip.weight*f(eq, ip, t)*get_detJ(eq, ip, t))
    end
    return sum(target)
end

function get_global_dofs(eq::Equation)
    eq.global_dofs
end

function set_global_dofs!(eq::Equation, dofs)
    eq.global_dofs = dofs
end

