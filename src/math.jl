# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

## This module contains math stuff, including interpolation, integration, linearization, ...

using ForwardDiff

#export interpolate, integrate, linearize

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
#function interpolate(e::Element, field::ASCIIString, x::Array{Float64,1}; derivative=false)
#    basis = derivative ? get_dbasisdxi(e) : get_basis(e)
#    return interpolate(e.attributes[field], basis, x)
#end



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
function linearize(f::Function, el::Element, field::ASCIIString)
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
    function jacobian(el::Element, args...)
        fld = get_field(el, field)
        dim, nnodes = size(fld)
        function helper!(x, y)
            orig = copy(fld)
            set_field(el, field,  reshape(x, dim, nnodes))
            y[:] = f(el, args...)
            set_field(el, field, copy(orig))
        end
        jac = ForwardDiff.forwarddiff_jacobian(helper!, Float64, fadtype=:dual, n=dim*nnodes, m=dim*nnodes)
        return jac(fld[:])
    end
    return jacobian
end

"""
In-place version, no additional garbage collection.
"""
function linearize!(f::Function, el::Element, field::ASCIIString, target::ASCIIString)
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
function integrate(f::Function, el::Element)
    target = []
    for ip in el.integration_points
        J = interpolate(el, "coordinates", ip.xi; derivative=true)
        push!(target, ip.weight*f(el, ip)*det(J))
    end
    return sum(target)
end
#function integrate(f::Function, integration_points::Array{IntegrationPoint, 1}, Xargs...)
#    target = []
#    for ip in integration_points
#        J = interpolate(el, "coordinates", ip.xi; derivative=true)
#        push!(target, ip.weight*f(ip, args...)*det(J))
#    end
#    return sum(target)
#end

"""
This version returns a function which must be operated with element e
"""
function integrate(f::Function)
    function integrate(el::Element)
        target = []
        for ip in el.integration_points
            J = interpolate(el, "coordinates", ip.xi; derivative=true)
            push!(target, ip.weight*f(el, ip)*det(J))
        end
        return sum(target)
    end
    return integrate
end

"""
This version saves results inplace to target, garbage collection free
"""
function integrate!(f::Function, el::Element, target)
    # set target to zero
    el.attributes[target][:] = 0.0
    for ip in el.integration_points
        J = interpolate(el, "coordinates", ip.xi; derivative=true)
        el.attributes[target][:,:] += ip.weight*f(el, ip)*det(J)
    end
end

get_integration_points(eq::Equation) = eq.integration_points

"""
Integrate f over element using Gaussian quadrature rules.

Parameters
----------
el::Element
    well defined element
f::Function
    Function to integrate
"""
function integrate(eq::Equation, f::Function)
    target = []
    for ip in get_integration_points(eq)
        J = get_jacobian(eq.element, ip.xi)
        push!(target, ip.weight*f(eq, ip)*det(J))
    end
    return sum(target)
end


"""
Evaluate field in point xi using basis functions.
"""
function interpolate(el::Element, field::ASCIIString, xi::Array{Float64,1})
    f = get_field(el, field)
    if !isa(f, Array)
        # This is scalar, nothing to interpolate
        return f
    end
    basis = get_basis(el, xi)
    dim, nnodes = size(f)
    result = zeros(dim)
    for i=1:nnodes
        result += basis[i]*f[:,i]
    end
    return result
end


function linearize(eq::Equation, f::Function, field::ASCIIString)
    function jacobian(eq::Equation, args...)
        el = get_element(eq)
        fld = get_field(el, field)
        dim, nnodes = size(fld)
        function helper(x::Vector)
            orig = copy(fld)
            set_field(el, field,  reshape(x, dim, nnodes))
            y = f(eq, args...)
            set_field(el, field, orig)
            return y[:]
        end
        jac = ForwardDiff.jacobian(helper)
        return jac(fld[:])
    end
    return jacobian
end


