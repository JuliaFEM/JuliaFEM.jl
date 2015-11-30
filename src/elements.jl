# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

abstract AbstractElement

type Element{E<:AbstractElement}
    connectivity :: Vector{Int}
    fields :: Dict{ASCIIString, Field}
end

function convert{E}(::Type{Element{E}}, connectivity::Vector{Int})
#   return Element{E}(connectivity, get_integration_points(E), Dict())
    return Element{E}(connectivity, Dict())
end

function get_integration_points{E}(element::Element{E})
    return get_integration_points(E)
end

function update_gauss_fields!(element::Element, data::Vector{IntegrationPoint}, time::Real)
    if haskey(element, "integration points")
        # push or update
        if !isapprox(last(element["integration points"]).time, time)
            push!(element["integration points"], time => data)
        else
            last(element["integration points"]).data = data
        end
    else
        # create
        element["integration points"] = Field(time => data)
    end
end

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
        element = Element{element_type}(collect(1:n))
    catch
        error("""
        Unable to create element with default constructor define function
        $eltype(connectivity) which initializes this element.""")
        return false
    end

    # try to interpolate some scalar field
    element["field1"] = range(1, n)
    # TODO: how to parametrize this?
    element["geometry"] = Vector{Float64}[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]

    # evaluate basis functions at middle point of element
    mid = zeros(dim)
    val1 = element(mid, 0.0)
    info("basis at $mid: $val1")
    val2 = element("field1", mid, 0.0)
    info("field val at $mid: $val2")
    val3 = element(mid, 0.0, Val{:grad})
    info("derivative of basis at $mid:\n$val3")
    #val4 = element("field1", mid, Val{:grad})
    #info("field val at $mid: $val4")

    info("Element $element_type passed tests.")
end

""" Get FieldSet from element. """
function Base.getindex(element::Element, field_name)
    return element.fields[field_name]
end

function Base.length{E}(element::Element{E})
    size(E)[2]
end

"""Add new Field to element.

Examples
--------
>>> element["temperature"] = [1, 2, 3, 4]
>>> element["temperature"] = (0.0, [0, 0, 0, 0]), (1.0, [1, 2, 3, 4])
>>> element["temperature"] = (0.0 => [0, 0, 0, 0], 1.0 => [1, 2, 3, 4])
"""
function Base.setindex!(element::Element, data, name::ASCIIString)
    element.fields[name] = Field(data)
end
function Base.setindex!(element::Element, field::Field, name::ASCIIString)
    element.fields[name] = field
end
function Base.setindex!(element::Element, data::Tuple, name::ASCIIString)
    element.fields[name] = Field(data...)
end

function get_connectivity(el::Element)
    return el.connectivity
end

typealias VecOrIP Union{Vector, IntegrationPoint}

function call(element::Element, field_name::ASCIIString, xi::VecOrIP, time::Number, variation=nothing)
    field = isa(variation, Void) ? element[field_name](time) : variation
    basis = get_basis(element)
    return basis(field, xi)
end

function call(element::Element, field_name::ASCIIString, xi::VecOrIP, time::Number, ::Type{Val{:grad}}, variation=nothing)
    field = isa(variation, Void) ? element[field_name](time) : variation
    basis = get_basis(element)
    geom = element["geometry"](time)
    return basis(geom, field, xi, Val{:grad})
end

function call(element::Element, field_name::ASCIIString, xi::VecOrIP)
    field = element[field_name]
    basis = get_basis(element)
    return basis(element[field_name], xi)
end

function call(element::Element, field_name::ASCIIString, xi::VecOrIP, ::Type{Val{:grad}})
    field = element[field_name]
    geom = element["geometry"]
    basis = get_basis(element)
    return basis(geom, field, xi, Val{:grad})
end

function call(element::Element, field_name::ASCIIString, time::Number)
    return element[field_name](time)
end

function get_dbasis{E<:AbstractElement}(::Type{E}, xi::Vector)
    basis(xi) = vec(get_basis(E, xi))
    return ForwardDiff.jacobian(basis, xi, cache=autodiffcache)'
end

function get_basis{E}(element::Element{E}, ip::IntegrationPoint)
    return get_basis(E, ip.xi)
end

function get_basis{E}(::Type{Element{E}}, xi::Vector{Float64})
    return get_basis(E, xi)
end

function get_basis{E}(element::Element{E}, xi::Vector{Float64})
    return get_basis(E, xi)
end

function call{E}(element::Element{E}, xi::VecOrIP, time::Float64=0.0)
    return get_basis(element, xi)
end

function get_basis{E}(element::Element{E})
    basis = CVTI(
        (xi::Vector) -> get_basis(E, xi),
        (xi::Vector) -> get_dbasis(E, xi))
    return basis
end

function call{E}(element::Element{E}, xi::VecOrIP, ::Type{Val{:grad}})
    basis = get_basis(element)
    geom = element["geometry"]
    return basis(geom, xi, Val{:grad})
end

function call{E}(element::Element{E}, xi::VecOrIP, time::Float64, ::Type{Val{:grad}})
    basis = get_basis(element)
    return basis(element["geometry"](time), xi, Val{:grad})
end

function call(element::Element, field_name::ASCIIString)
    return element[field_name]
end

function LinAlg.det{E<:AbstractElement}(element::Element{E}, ip::IntegrationPoint, time::Number=0.0)
    X = element("geometry", time)
    dN = get_dbasis(E, ip.xi)
    J = sum([dN[:,i]*X[i]' for i=1:length(X)])
    m, n = size(J)
    return m == n ? det(J) : norm(J)
end

""" Check does field exist. """
function Base.haskey(element::Element, what)
    haskey(element.fields, what)
end

