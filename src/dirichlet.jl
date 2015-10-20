# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# Dirichlet boundary conditions in weak form

abstract DirichletEquation <: Equation

get_unknown_field_name(eq::DirichletEquation) = symbol("reaction force")

### Dirichlet problem + equations

type DirichletProblem <: BoundaryProblem
    equations :: Array{DirichletEquation, 1}
end
function DirichletProblem()
    DirichletProblem([])
end
get_dimension(pr::Type{DirichletProblem}) = 1  # ..?
get_equation(pr::Type{DirichletProblem}, el::Type{Seg2}) = DBC2D2

"""
Dirichlet boundary condition element for 2 node line segment
"""
type DBC2D2 <: DirichletEquation
    element :: Seg2
    integration_points :: Array{IntegrationPoint, 1}
    global_dofs :: Array{Int64, 1}
    fieldval :: Function
end
function DBC2D2(element::Seg2)
    integration_points = [
        IntegrationPoint([-sqrt(1/3)], 1.0),
        IntegrationPoint([+sqrt(1/3)], 1.0)]
    push!(element, FieldSet("reaction force"))
    fieldval(X, t) = 0.0
    DBC2D2(element, integration_points, [], fieldval)
end
function get_lhs(eq::DBC2D2, ip, t)
    el = get_element(eq)
    h = get_basis(el)(ip.xi)
    return h*h'
end
function get_rhs(eq::DBC2D2, ip, t)
    el = get_element(eq)
    h = get_basis(el, ip.xi)
    f = eq.fieldval
    X = interpolate(el, "geometry", ip.xi, t)
    return h*f(X, t)
end
has_lhs(eq::DBC2D2) = true
has_rhs(eq::DBC2D2) = true

