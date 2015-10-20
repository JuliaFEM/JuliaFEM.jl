# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# Heat problems

abstract HeatProblem <: Problem
abstract HeatEquation <: Equation

get_unknown_field_name(eq::HeatEquation) = symbol("temperature")


### Plane heat problem + equations ###

type PlaneHeatProblem <: HeatProblem
    equations :: Array{HeatEquation, 1}
end

""" Default constructor for problem takes no arguments. """
function PlaneHeatProblem()
    return PlaneHeatProblem([])
end

""" Return dimension of unknown field variable, temperature is scalar field. """
get_dimension(pr::Type{PlaneHeatProblem}) = 1

""" Map Lagrange element Quad4 to equation DC2D4 """
get_equation(pr::Type{PlaneHeatProblem}, el::Type{Quad4}) = DC2D4

""" Map Lagrange element Seg2 to equation DC2D2 """
get_equation(pr::Type{PlaneHeatProblem}, el::Type{Seg2}) = DC2D2


""" Diffusive heat transfer for 4-node bilinear element. """
type DC2D4 <: HeatEquation
    element :: Quad4
    integration_points :: Array{IntegrationPoint, 1}
    global_dofs :: Array{Int64, 1}
end
function DC2D4(element::Quad4)
    integration_points = [
        IntegrationPoint(1.0/sqrt(3.0)*[-1, -1], 1.0),
        IntegrationPoint(1.0/sqrt(3.0)*[ 1, -1], 1.0),
        IntegrationPoint(1.0/sqrt(3.0)*[ 1,  1], 1.0),
        IntegrationPoint(1.0/sqrt(3.0)*[-1,  1], 1.0)]
    push!(element, FieldSet("temperature"))
    DC2D4(element, integration_points, [])
end
function get_lhs(equation::DC2D4, ip, time)
    element = get_element(equation)
    dNdX = get_dbasisdX(element, ip.xi, time)
    k = interpolate(element, "temperature thermal conductivity", ip.xi, time)
    return dNdX*k*dNdX'
end
JuliaFEM.has_lhs(eq::DC2D4) = true

""" Diffusive heat transfer for 2-node linear segment. """
type DC2D2 <: HeatEquation
    element :: Seg2
    integration_points :: Array{IntegrationPoint, 1}
    global_dofs :: Array{Int64, 1}
end
function DC2D2(element::Seg2)
    integration_points = [IntegrationPoint([0.0], 2.0)]
    push!(element, FieldSet("temperature"))
    DC2D2(element, integration_points, [])
end
function get_rhs(equation::DC2D2, ip, time)
    element = get_element(equation)
    h = get_basis(element, ip.xi)
    f = interpolate(element, "temperature flux", ip.xi, time)
    return h*f
end
JuliaFEM.has_rhs(eq::DC2D2) = true
