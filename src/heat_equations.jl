# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

abstract Heat <: Equation

"""
Diffusive heat transfer for 4-node bilinear element.
"""
type DC2D4 <: Heat
    element :: Quad4
    integration_points :: Array{IntegrationPoint, 1}
    global_dofs :: Array{Int64, 1}
end
function DC2D4(el::Quad4)
    integration_points = [
        IntegrationPoint(1.0/sqrt(3.0)*[-1, -1], 1.0),
        IntegrationPoint(1.0/sqrt(3.0)*[ 1, -1], 1.0),
        IntegrationPoint(1.0/sqrt(3.0)*[ 1,  1], 1.0),
        IntegrationPoint(1.0/sqrt(3.0)*[-1,  1], 1.0)]
    set_field(el, :temperature, zeros(2, 4))
    DC2D4(el, integration_points, [])
end
function get_lhs(eq::DC2D4)
    function get_lhs_(eq, ip)
        el = get_element(eq)
        xi = ip.xi
        dNdX = get_dbasisdX(el, xi)'
        hc = interpolate(el, :"temperature heat coefficient", xi)
        return dNdX'*hc*dNdX
    end
    integrate(eq, get_lhs_)
end


"""
Diffusive heat transfer for 2-node linear segment.
"""
type DC2D2 <: Heat
    element :: Seg2
    integration_points :: Array{IntegrationPoint, 1}
    global_dofs :: Array{Int64, 1}
end
function DC2D2(el::Seg2)
    integration_points = [IntegrationPoint([0.0], 1.0)]
    set_field(el, :temperature, zeros(2, 1))
    set_field(el, :"temperature flux", zeros(2, 1))
    DC2D2(el, integration_points, [])
end
function get_rhs(eq::DC2D2)
    function get_rhs_(eq, ip)
        el = get_element(eq)
        xi = ip.xi
        N = get_basis(el, xi)
        f = interpolate(el, :"temperature flux", xi)
        return f*N
    end
    integrate(eq, get_rhs_)
end