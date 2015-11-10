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
    info("Testing element $element_type")
    local element
    dim = nothing
    n = nothing
    try
        dim, n = size(element_type)
    catch
        error("Unable to determine element dimensions. Define Base.size(element::Type{$elementtype}) = (dim, nbasis) where dim is spatial dimension of element and nbasis is number of basis functions of element.")
    end
    info("element dimension: $dim x $n")

    info("Initializing element")
    try
        element = element_type(collect(1:n))
    catch
        error("""
        Unable to create element with default constructor define function
        $eltype(connectivity) which initializes this element.""")
        return false
    end

    # try to interpolate some scalar field
    element["field1"] = Field(collect(1:n))
    # TODO: how to parametrize this?
    element["geometry"] = Field(Vector[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

    # evaluate basis functions at middle point of element
    basis = get_basis(element)
    dbasis = grad(basis)
    mid = zeros(dim)
    val1 = basis(mid, 0.0)
    info("basis at $mid: $val1")
    val2 = basis("field1", mid, 0.0)
    info("field val at $mid: $val2")
    val3 = dbasis(mid, 0.0)
    info("derivative of basis at $mid:\n$val3")
    val4 = dbasis("field1", mid, 0.0)
    info("field val at $mid: $val4")

    info("Element $element_type passed tests.")
end

""" Get FieldSet from element. """
function Base.getindex(element::Element, field_name)
    return element.fields[field_name]
end

"""Add new Field to element.

Examples
--------
>>> element["temperature"] = [1, 2, 3, 4]
>>> element["temperature"] = (0.0, [0, 0, 0, 0]), (1.0, [1, 2, 3, 4])
>>> element["temperature"] = (0.0 => [0, 0, 0, 0], 1.0 => [1, 2, 3, 4])
"""
function Base.setindex!(element::Element, field_data, field_name)
    setindex!(element.fields, field_data, field_name)
end

function Base.setindex!(element::Element, field_data::Tuple, field_name)
    field = Field()
    for (time, data) in field_data
        ts = TimeStep(time, Increment[Increment(data)])
        push!(field, ts)
    end
    element[field_name] = field
end

function get_connectivity(el::Element)
    return el.connectivity
end

abstract AbstractFunctionSpace

type FunctionSpace <: AbstractFunctionSpace
    basis :: Basis
    fields :: FieldSet
end

type GradientFunctionSpace <: AbstractFunctionSpace
    basis :: Basis
    fields :: FieldSet
end

function get_basis(element::Element)
    return FunctionSpace(element.basis, element.fields)
end

function get_dbasis(element::Element)
    return GradientFunctionSpace(element.basis, element.fields)
end

function grad(u::FunctionSpace)
    return GradientFunctionSpace(u.basis, u.fields)
end

""" If basis is called without a field, return basis functions evaluated at that point. """
function call(u::FunctionSpace, xi::Union{Vector, IntegrationPoint}, t::Number=0.0)
    return u.basis(xi)
end

""" If gradient of basis is called without a field, return "empty" gradient evaluated at that point. """
function call(gradu::GradientFunctionSpace, xi::Union{Vector, IntegrationPoint}, t::Number=0.0)
    geometry = gradu.fields["geometry"](t)
    gradu.basis(geometry, xi, Val{:grad})
end

""" Evaluate field on element function space. """
function call(u::FunctionSpace, field_name, xi::Union{Vector, IntegrationPoint}, t::Number=0.0, variation=nothing)
    field = !isa(variation, Void) ? variation : u.fields[field_name](t)
    if length(field) == 1
        return field.data[1]
    end
    u.basis(field, xi)
end

""" Evaluate gradient of field on element function space. """
function call(gradu::GradientFunctionSpace, field_name, xi::Union{Vector, IntegrationPoint}, t::Number=0.0, variation=nothing)
    field = !isa(variation, Void) ? variation : gradu.fields[field_name](t)
    geometry = gradu.fields["geometry"](t)
    gradu.basis(geometry, field, xi, Val{:grad})
end


# on-line functions to get api more easy to use, ip -> xi.ip
#call(u::FunctionSpace, ip::IntegrationPoint, t::Number=Inf) = call(u, ip.xi, t)
#call(u::GradientFunctionSpace, ip::IntegrationPoint, t::Number=Inf) = call(u, ip.xi, t)
# i think these will be the most called functions.
#call(u::FunctionSpace, field_name, ip::IntegrationPoint, t::Number=0.0, variation=nothing) = call(u, field_name, ip.xi, t, variation)
#call(u::GradientFunctionSpace, field_name, ip::IntegrationPoint, t::Number=0.0, variation=nothing) = call(u, field_name, ip.xi, t, variation)
#call(u::FunctionSpace, field_name) = (args...) -> call(u, field_name, args...)
#call(u::GradientFunctionSpace, field_name) = (args...) -> call(u, field_name, args...)

""" Return a field from function space. """
function get_field(u::FunctionSpace, field_name, time::Number=0.0)
    return u.fields[field_name](time)
end

""" Return a field from function space. """
function get_field(u::FunctionSpace, field_name, time::Number=0.0, variation=nothing)
    return !isa(variation, Void) ? variation : u.fields[field_name](time)
end

""" Return a field from function space. """
function get_fieldset(u::FunctionSpace, field_name)
    return u.fields[field_name]
end

""" Get a determinant of element in point ξ. """
function LinAlg.det(u::FunctionSpace, xi::Vector, time::Number=0.0)
    X = u.fields["geometry"](time)
    dN = u.basis.dbasisdxi(xi)
    J = sum([dN[:,i]*X[i]' for i=1:length(X)])
    m, n = size(J)
    return m == n ? det(J) : norm(J)
end
function LinAlg.det(u::FunctionSpace, ip::IntegrationPoint, time::Number=0.0)
    LinAlg.det(u, ip.xi, time)
end
function LinAlg.det(u::FunctionSpace)
    return (args...) -> det(u, args...)
end
#Base.(:+)(u::FunctionSpace, v::FunctionSpace) = (args...) -> u(args...) + v(args...)
#Base.(:-)(u::FunctionSpace, v::FunctionSpace) = (args...) -> u(args...) - v(args...)
#Base.(:+)(u::GradientFunctionSpace, v::GradientFunctionSpace) = (args...) -> u(args...) + v(args...)
#Base.(:-)(u::GradientFunctionSpace, v::GradientFunctionSpace) = (args...) -> u(args...) - v(args...)

""" Check does field exist. """
function Base.haskey(element::Element, what)
    haskey(element.fields, what)
end

