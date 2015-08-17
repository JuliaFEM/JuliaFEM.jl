# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
This module contains math stuff, including interpolation, integration, linearization, ...
"""

using JuliaFEM
using ForwardDiff

"""
Interpolate field variable using basis functions f for point ip.
This function tries to be as general as possible and allows interpolating
lot of different fields.

Parameters
----------
field :: Array{Number, dim}
  Field variable
basis :: Function
  Basis functions
ip :: Array{Number, 1}
  Point to interpolate
"""
function interpolate(field::Float64, basis::Function, ip::Array{Float64,1})
    # dummy function, unable to interpolate scalar value!
    return field
end
function interpolate{T<:Real}(field::Array{T,1}, basis::Function, ip)
    result = dot(field, basis(ip))
    return result
end
function interpolate{T<:Real}(field::Array{T,2}, basis::Function, ip)
    m, n = size(field)
    bip = basis(ip)
    tmp = size(bip)
    if length(tmp) == 1
        ndim = 1
        nnodes = tmp[1]
    else
        ndim, nnodes = size(bip)
    end
    if ndim == 1
        if n == nnodes
            result = field * bip
        elseif m == nnodes
            result = field' * bip
        end
    else
      if n == nnodes
        result = bip' * field
    elseif m == nnodes
      result = bip' * field'
    end
    end
    if length(result) == 1
        result = result[1]
    end
    return result
end
function interpolate(e::Element, field::ASCIIString, x::Array{Float64,1}; derivative=false)
    return interpolate(e.attributes[field], derivative ? e.dbasis : e.basis, x)
end


"""

"""
function get_basis(el::Element, xi)
    return el.basis(xi)
end

"""
Return partial derivatives of shape functions w.r.t X using chain rule.
"""
function get_dbasisdX(el::Element, xi)
    J = interpolate(el, "coordinates", xi; derivative=true)
    dbasisdX = el.dbasis(xi)*inv(J')
    return dbasisdX
end


"""
Linearize function f w.r.t some given field, i.e. calculate dR/du

Parameters
----------
f::Function
    (possibly) nonlinear function to linearize
field::ASCIIString
    field variable

Returns
-------
Array{Float64, 2}
    jacobian / "tangent stiffness matrix"

"""
function linearize(f::Function, el::JuliaFEM.Element, field::ASCIIString)
    dim, nnodes = size(el.attributes[field])
    function helper!(x, y)
        orig = copy(el.attributes[field])
        el.attributes[field] = reshape(x, dim, nnodes)
        y[:] = f(el)
        el.attributes[field] = copy(orig)
    end
    jac = ForwardDiff.forwarddiff_jacobian(helper!, Float64, fadtype=:dual, n=dim*nnodes, m=dim*nnodes)
    return jac(el.attributes[field][:])
end

"""
This version returns another function which can be then evaluated against field
"""
function linearize(f::Function, field::ASCIIString)
    function jacobian(el::JuliaFEM.Element, args...)
        dim, nnodes = size(el.attributes[field])
        function helper!(x, y)
            orig = copy(el.attributes[field])
            el.attributes[field] = reshape(x, dim, nnodes)
            y[:] = f(el, args...)
            el.attributes[field] = copy(orig)
        end
        jac = ForwardDiff.forwarddiff_jacobian(helper!, Float64, fadtype=:dual, n=dim*nnodes, m=dim*nnodes)
        return jac(el.attributes[field][:])
    end
    return jacobian
end

"""
In-place version, no additional garbage collection.
"""
function linearize!(f::Function, el::JuliaFEM.Element, field::ASCIIString, target::ASCIIString)
    el.attributes[target][:] = 0.0
    dim, nnodes = size(el.attributes[field])
    function helper!(x, y)
        orig = copy(el.attributes[field])
        el.attributes[field] = reshape(x, dim, nnodes)
        y[:] = f(el)
        el.attributes[field] = copy(orig)
    end
    jac! = ForwardDiff.forwarddiff_jacobian!(helper!, Float64, fadtype=:dual, n=dim*nnodes, m=dim*nnodes)
    jac!(el.attributes[field][:], el.attributes[target])
end



"""
Integrate f over element using Gaussian quadrature rules.

Parameters
----------
el::Element
    well defined element
f::Function
    Function to integrate
"""
function integrate(f::Function, el::JuliaFEM.Element)
    target = []
    for m = 1:length(el.iweights)
        w = el.iweights[m]
        xi = el.ipoints[:, m]
        J = JuliaFEM.interpolate(el, "coordinates", xi; derivative=true)
        push!(target, w*f(el, xi)*det(J))
    end
    return sum(target)
end

"""
This version returns a function which must be operated with element e
"""
function integrate(f::Function)
    function integrate(el::JuliaFEM.Element)
        target = []
        for m = 1:length(el.iweights)
            w = el.iweights[m]
            xi = el.ipoints[:, m]
            J = JuliaFEM.interpolate(el, "coordinates", xi; derivative=true)
            push!(target, w*f(el, xi)*det(J))
        end
        return sum(target)
    end
    return integrate
end

"""
This version saves results inplace to target, garbage collection free
"""
function integrate!(f::Function, el::JuliaFEM.Element, target)
    # set target to zero
    el.attributes[target][:] = 0.0
    for m = 1:length(el.iweights)
        w = el.iweights[m]
        xi = el.ipoints[:, m]
        J = JuliaFEM.interpolate(el, "coordinates", xi; derivative=true)
        el.attributes[target][:,:] += w*f(el, xi)*det(J)
    end
end
