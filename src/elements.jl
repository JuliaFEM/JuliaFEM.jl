# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

abstract CG <: Element # Lagrange element family
abstract Quad4 <: CG # 4 node quadrangle elements


"""
Evaluate basis functions in point xi.
"""
function get_basis(el::Quad4)
    (xi) -> [(1-xi[1])*(1-xi[2])/4
             (1+xi[1])*(1-xi[2])/4
             (1+xi[1])*(1+xi[2])/4
             (1-xi[1])*(1+xi[2])/4]
end
function get_basis(el::Quad4, xi)
    get_basis(el)(xi)
end

"""
Evaluate partial derivatives of basis function w.r.t
dimensionless coordinate xi, i.e. dbasis/dxi
"""
function get_dbasisdxi(el::Quad4)
    (xi) -> [-(1-xi[2])/4.0    -(1-xi[1])/4.0
              (1-xi[2])/4.0    -(1+xi[1])/4.0
              (1+xi[2])/4.0     (1+xi[1])/4.0
             -(1+xi[2])/4.0     (1-xi[1])/4.0]
end
function get_dbasisdxi(el::Quad4, xi)
    get_dbasisdxi(el)(xi)
end



"""
Get jacobian of element evaluated at point xi
"""
function get_jacobian(el::Element, xi)
    dbasisdxi = get_dbasisdxi(el)
    X = get_coordinates(el)
    J = interpolate(X, dbasisdxi, xi)'
    return J
end

"""
Evaluate partial derivatives of basis function w.r.t
material description X, i.e. dbasis/dX
"""
function get_dbasisdX(el::CG)
    function get_dbasisdX_(xi)
        dbasisdxi = get_dbasisdxi(el, xi)
        J = get_jacobian(el, xi)
        dbasisdxi*inv(J)
    end
end
function get_dbasisdX(el::CG, xi)
    get_dbasisdX(el)(xi)
end

"""
Return coordinates of element in array of size dim x nnodes
"""
function get_coordinates(el::Element)
    # Make sure you define at least this field to your element if you want
    # to build everything yourself
    el.coordinates
end

function set_coordinates(el::Element, coordinates)
    el.coordinates = coordinates
end

function set_material(el::Element, lambda, mu)
    el.attributes["lambda"] = lambda
    el.attributes["mu"] = mu
end

"""
Get element id
"""
function get_element_id(el::Element)
    el.id
end

function get_integration_points(el::Element)
    el.integration_points
end



