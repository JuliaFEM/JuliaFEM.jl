# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using ForwardDiff

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
This version returns a function which must be operated with element e
"""
function integrate(f::Function)
    function integrate(el::Element)
        target = []
        for ip in el.integration_points
            J = interpolate(el, :geometry, ip.xi; derivative=true)
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
        J = interpolate(el, :geometry, ip.xi; derivative=true)
        el.attributes[target][:,:] += ip.weight*f(el, ip)*det(J)
    end
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

