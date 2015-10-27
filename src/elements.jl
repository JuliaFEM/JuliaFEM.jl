# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

#=
Related notebooks
-----------------

2015-08-29-developing-juliafem.ipynb
=#

using FactCheck
using ForwardDiff

abstract Element

""" Get FieldSet from element. """
function Base.getindex(element::Element, field_name)
    element.fields[field_name]
end

""" Add new FieldSet to element. """
function Base.setindex!(element::Element, fieldset::FieldSet, fieldset_name)
    fieldset.name = fieldset_name
    element.fields[fieldset.name] = fieldset
end
function Base.push!(element::Element, fieldset::FieldSet)
    element[fieldset.name] = fieldset
end


#= ELEMENT DEFINITIONS

Example
-------

This is example how to create new element. This is commented because I use code
generation for simple elements like Lagrage elements. Feel free to use
code generation but elements can be of course created manually too!

abstract CG <: Element # create new element family "Continous Galerkin"

type Quad4 <: CG
    connectivity :: Array{Int, 1}
    fields :: Dict{Any, Any}
end

""" Default contructor. """
Quad4(connectivity) = Quad4(connectivity, Dict{Any, Any}())

""" Return number of basis functions of this element. """
get_number_of_basis_functions(el::Type{Quad4}) = 4

""" Return element dimension (length of xi vector). """
get_element_dimension(el::Type{Quad4}) = 2

""" Return basis functions for this element (xi dim = 2, functions = 4). """
function get_basis(el::Quad4, xi)
    [(1-xi[1])*(1-xi[2])/4
     (1+xi[1])*(1-xi[2])/4
     (1+xi[1])*(1+xi[2])/4
     (1-xi[1])*(1+xi[2])/4]
end

""" Return partial derivatives of basis functions. """
function get_dbasisdxi(el::Quad4, xi)
    [-(1-xi[2])/4.0    -(1-xi[1])/4.0
      (1-xi[2])/4.0    -(1+xi[1])/4.0
      (1+xi[2])/4.0     (1+xi[1])/4.0
     -(1+xi[2])/4.0     (1-xi[1])/4.0]
end

End of example.

=#

# define size of your element as (dim, nbasis) tuple where first integer is spatial dimension and second is number of basis functions.
# Base.size(element::Type{Element}) = nothing

### COMMON ELEMENT ROUTINES ###

"""
Test routine for element. If this passes, element interface is properly
defined.

Parameters
----------
eltype::Type{Element}
    Element to test

Raises
------
This uses FactCheck and throws exceptions if element is not passing all tests.
"""
function test_element(element_type)
    Logging.info("Testing element $element_type")
    local element
    dim = nothing
    n = nothing
    try
        dim, n = size(element_type)
    catch
        Logging.error("Unable to determine element dimensions. Define Base.size(element::Type{$elementtype}) = (dim, nbasis) where dim is spatial dimension of element and nbasis is number of basis functions of element.")
    end
    Logging.info("element dimension: $dim x $n")

    Logging.info("Initializing element")
    try
        element = element_type(collect(1:n))
    catch
        Logging.error("""
        Unable to create element with default constructor define function
        $eltype(connectivity) which initializes this element.""")
        return false
    end

    # try to interpolate some scalar field
    push!(element, FieldSet("field1", [Field(0.0, collect(1:n))]))
    # TODO: how to parametrize this?
    push!(element, FieldSet("geometry", [Field(0.0, Vector[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])]))

    # evaluate basis functions at middle point of element
    basis = get_basis(element)
    dbasis = grad(basis)
    mid = zeros(dim)
    val1 = basis(mid, 0.0)
    Logging.info("basis at $mid: $val1")
    val2 = basis("field1", mid, 0.0)
    Logging.info("field val at $mid: $val2")
    val3 = dbasis(mid, 0.0)
    Logging.info("derivative of basis at $mid: $val3")
    val4 = dbasis("field1", mid, 0.0)
    Logging.info("field val at $mid: $val4")

    Logging.info("Element $element_type passed tests.")
end


function get_connectivity(el::Element)
    el.connectivity
end

type MixedFunctionSpace
    element1 :: Element
    element2 :: Element
end

type FunctionSpace
    element :: Element
end

type GradientFunctionSpace
    element :: Element
end

function grad(u::FunctionSpace)
    GradientFunctionSpace(u.element)
end

""" Evaluate field on element function space. """
function call(u::FunctionSpace, field_name, xi::Vector, t::Number=Inf, variation=nothing)
    f = !isa(variation, Void) ? variation : u.element[field_name](t)
    if length(f) == 1
        return f.values
    end
    h = u.element.basis.basis(xi)
    return dot(vec(h), f)
end

""" If basis is called without a field, return basis functions evaluated at that point. """
function call(u::FunctionSpace, xi::Vector, t::Number=Inf)
    return u.element.basis.basis(xi)
end

""" Evaluate gradient of field on element function space. """
function call(gradu::GradientFunctionSpace, field_name, xi::Vector, t::Number=Inf, variation=nothing)
    f = !isa(variation, Void) ? variation : gradu.element[field_name](t)
    X = gradu.element["geometry"](t)
    dN = gradu.element.basis.dbasisdxi(xi)
    J = sum([dN[:,i]*X[i]' for i=1:length(X)])
    grad = inv(J)*dN
    gradf = sum([grad[:,i]*f[i]' for i=1:length(f)])'
    return gradf
end

""" If gradient of basis is called without a field, return "empty" gradient evaluated at that point. """
function call(gradu::GradientFunctionSpace, xi::Vector, t::Number=Inf)
    X = gradu.element["geometry"](t)
    dN = gradu.element.basis.dbasisdxi(xi)
    J = sum([dN[:,i]*X[i]' for i=1:length(X)])
    grad = inv(J)*dN
    return grad
end

# on-line functions to get api more easy to use, ip -> xi.ip
call(u::FunctionSpace, ip::IntegrationPoint, t::Number=Inf) = call(u, ip.xi, t)
call(u::GradientFunctionSpace, ip::IntegrationPoint, t::Number=Inf) = call(u, ip.xi, t)
# i think these will be the most called functions.
call(u::FunctionSpace, field_name, ip::IntegrationPoint, t::Number, variation=nothing) = call(u, field_name, ip.xi, t, variation)
call(u::GradientFunctionSpace, field_name, ip::IntegrationPoint, t::Number, variation=nothing) = call(u, field_name, ip.xi, t, variation)
call(u::FunctionSpace, field_name) = (args...) -> call(u, field_name, args...)
call(u::GradientFunctionSpace, field_name) = (args...) -> call(u, field_name, args...)

""" Return a field from function space. """
function get_field(u::FunctionSpace, field_name, time=Inf)
    return u.element[field_name](time)
end

""" Return a field from function space. """
function get_field(u::FunctionSpace, field_name, time=Inf, variation=nothing)
    return !isa(variation, Void) ? variation : u.element[field_name](time)
end

""" Return a fieldset from function space. """
function get_fieldset(u::FunctionSpace, field_name)
    return u.element[field_name]
end

function LinAlg.det(u::FunctionSpace, xi::Vector, t::Number=Inf)
    X = u.element["geometry"](t)
    dN = u.element.basis.dbasisdxi(xi)
    J = sum([dN[:,i]*X[i]' for i=1:length(X)])
    m, n = size(J)
    return m == n ? det(J) : norm(J)
end
function LinAlg.det(u::FunctionSpace, ip::IntegrationPoint, t::Number=Inf)
    LinAlg.det(u, ip.xi, t)
end
function LinAlg.det(u::FunctionSpace)
    return (args...) -> det(u, args...)
end

function get_basis(element::Element)
    return FunctionSpace(element)
end

Base.(:+)(u::FunctionSpace, v::FunctionSpace) = (args...) -> u(args...) + v(args...)
Base.(:-)(u::FunctionSpace, v::FunctionSpace) = (args...) -> u(args...) - v(args...)
Base.(:+)(u::GradientFunctionSpace, v::GradientFunctionSpace) = (args...) -> u(args...) + v(args...)
Base.(:-)(u::GradientFunctionSpace, v::GradientFunctionSpace) = (args...) -> u(args...) - v(args...)


""" Check does fieldset exist. """
function Base.haskey(element::Element, what)
    haskey(element.fields, what)
end


# FIXME: These two needs integration -- maybe not in elements.jl ..?
"""
Fit field s.t. || ∫ (Nᵢ(ξ)αᵢ - f(el, ξ)) dS || -> min!

Parameters
----------
f::Function
    Needs to take (el::Element, xi::Vector) as argument
fixed_coeffs::Int[]
    These coefficients are not changed during fitting -> constrained optimizatio
"""
function fit_field!(el::Element, field, f, fixed_coeffs=Int[])
    w = [
        128/225,
        (332+13*sqrt(70))/900,
        (332+13*sqrt(70))/900,
        (332-13*sqrt(70))/900,
        (332-13*sqrt(70))/900]
    xi = Vector[
        [0.0],
        [ 1/3*sqrt(5 - 2*sqrt(10/7))],
        [-1/3*sqrt(5 - 2*sqrt(10/7))],
        [ 1/3*sqrt(5 + 2*sqrt(10/7))],
        [-1/3*sqrt(5 + 2*sqrt(10/7))]]
    n = get_number_of_basis_functions(el)
    fld = get_field(el, field)
    nfld = length(fld[1])
    #Logging.debug("dim of field $field: $nfld")

    M = zeros(n, n)
    b = zeros(n, nfld)
    for i=1:length(w)
        detJ = get_detJ(el, xi[i])
        N = get_basis(el, xi[i])
        M += w[i]*N*N'*detJ
        fi = f(el, xi[i])
        for j=1:nfld
            b[:, j] += w[i]*N*fi[j]*detJ
        end
    end

    coeffs = zeros(n)
    for j=1:nfld
        for k=1:n
            coeffs[k] = fld[k][j]
        end
        if length(fixed_coeffs) != 0
            # constrained problem, some coefficients are fixed
            N = Int[] # rest of coeffs
            S = Int[] # fixed coeffs
            for i = 1:n
                if i in fixed_coeffs
                    push!(S, i)
                else
                    push!(N, i)
                end
            end
            lhs = M[N,N]
            rhs = b[N,j] - M[N,S]*coeffs[S]
            coeffs[N] = lhs \ rhs
        else
            coeffs[:] = M \ b[:,j]
        end
        for k=1:n
            fld[k][j] = coeffs[k]
        end
    end
    set_field(el, field, fld)
    return
end


"""
Fit field s.t. || ∫ ∂/∂ξ(∑Nᵢ(ξ)αᵢ)f(el, ξ) dS || -> min!
"""
function fit_derivative_field!(el::Element, field, f, fixed_coeffs=Int[])
    w = [
        128/225,
        (332+13*sqrt(70))/900,
        (332+13*sqrt(70))/900,
        (332-13*sqrt(70))/900,
        (332-13*sqrt(70))/900]
    xi = Vector[
        [0.0],
        [ 1/3*sqrt(5 - 2*sqrt(10/7))],
        [-1/3*sqrt(5 - 2*sqrt(10/7))],
        [ 1/3*sqrt(5 + 2*sqrt(10/7))],
        [-1/3*sqrt(5 + 2*sqrt(10/7))]]
    n = get_number_of_basis_functions(el)
    fld = get_field(el, field)
    nfld = length(fld[1])
    #Logging.debug("dim of field $field: $nfld")

    M = zeros(n, n)
    b = zeros(n, nfld)
    for i=1:length(w)
        detJ = get_detJ(el, xi[i])
        dNdxi = get_dbasisdxi(el, xi[i])
        dNdX = dNdxi / detJ
        M += w[i]*dNdX*dNdX'*detJ
        fi = f(el, xi[i])
        for j=1:nfld
            b[:, j] += w[i]*dNdX*fi[j]*detJ
        end
    end

    coeffs = zeros(n)
    for j=1:nfld
        for k=1:n
            coeffs[k] = fld[k][j]
        end
        if length(fixed_coeffs) != 0
            #Logging.info("constrained problem, some coefficients are fixed")
            N = Int[] # rest of coeffs
            S = Int[] # fixed coeffs
            for i = 1:n
                if i in fixed_coeffs
                    push!(S, i)
                else
                    push!(N, i)
                end
            end
            lhs = M[N,N]
            rhs = b[N,j] - M[N,S]*coeffs[S]
            coeffs[N] = lhs \ rhs
        else
            coeffs[:] = M \ b[:,j]
        end
        for k=1:n
            fld[k][j] = coeffs[k]
        end
    end
    set_field(el, field, fld)
    return
end


