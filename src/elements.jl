# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using FactCheck
using ForwardDiff

abstract Element

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
    element["field1"] = Field(0.0, collect(1:n))
    # TODO: how to parametrize this?
    element["geometry"] = Field(0.0, Vector[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

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

""" Get FieldSet from element. """
function Base.getindex(element::Element, field_name)
    element.fields[field_name]
end

"""Add new FieldSet to element.

Examples
--------
>>> element["geometry"] = [1, 2, 3, 4]
JuliaFEM.Quad4([1,2,3,4],JuliaFEM.Basis(basis,dbasisdxi),Dict("geometry"=>JuliaFEM.FieldSet("geometry",JuliaFEM.Field[JuliaFEM.Field{Array{Int64,1}}(0.0,0,[1,2,3,4])])))
"""
function Base.setindex!(element::Element, field_data, field_name)
    element.fields[field_name] = field_data
end

function get_connectivity(el::Element)
    el.connectivity
end

abstract AbstractFunctionSpace

type FunctionSpace <: AbstractFunctionSpace
    element :: Element
end

type GradientFunctionSpace <: AbstractFunctionSpace
    element :: Element
end

type MixedFunctionSpace <: AbstractFunctionSpace
    element1 :: Element
    element2 :: Element
end

function get_basis(element::Element)
    return FunctionSpace(element)
end

function get_dbasis(element::Element)
    return GradientFunctionSpace(element)
end

function grad(u::FunctionSpace)
    return GradientFunctionSpace(u.element)
end

""" Evaluate field on element function space. """
function call(u::FunctionSpace, field_name, xi::Vector, t::Number=Inf, variation=nothing)
    f = !isa(variation, Void) ? variation : u.element[field_name](t)
    if length(f) == 1
        return f.data[1]
    end
    h = u.element.basis.basis(xi)
    #@debug("vec(h) = $(vec(h)), size(h) = $(size(vec(h)))")
    #@debug("f = $f, size(f) = $(size(f))")
    #return dot(vec(h), f)
    return sum(vec(h).*f)
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
call(u::FunctionSpace, field_name, ip::IntegrationPoint, t::Number=Inf, variation=nothing) = call(u, field_name, ip.xi, t, variation)
call(u::GradientFunctionSpace, field_name, ip::IntegrationPoint, t::Number=Inf, variation=nothing) = call(u, field_name, ip.xi, t, variation)
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

""" Get a determinant of element in point ξ. """
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

Base.(:+)(u::FunctionSpace, v::FunctionSpace) = (args...) -> u(args...) + v(args...)
Base.(:-)(u::FunctionSpace, v::FunctionSpace) = (args...) -> u(args...) - v(args...)
Base.(:+)(u::GradientFunctionSpace, v::GradientFunctionSpace) = (args...) -> u(args...) + v(args...)
Base.(:-)(u::GradientFunctionSpace, v::GradientFunctionSpace) = (args...) -> u(args...) - v(args...)

""" Check does fieldset exist. """
function Base.haskey(element::Element, what)
    haskey(element.fields, what)
end

