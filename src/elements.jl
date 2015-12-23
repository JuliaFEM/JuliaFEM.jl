# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

abstract AbstractElement

type Element{E<:AbstractElement}
    connectivity :: Vector{Int}
    fields :: Dict{ASCIIString, Field}
end

function Base.size{E}(::Element{E})
    return size(E)
end

function Base.size{E}(::Element{E}, i::Int64)
    return size(E)[i]
end

function convert{E}(::Type{Element{E}}, connectivity::Vector{Int})
#   return Element{E}(connectivity, get_integration_points(E), Dict())
    return Element{E}(connectivity, Dict())
end

function get_integration_points{E}(element::Element{E}, args...)
    return get_integration_points(E, args...)
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

function call(element::Element, field_name::ASCIIString, time::Real, variation=nothing)
    return isa(variation, Void) ? element[field_name](time) : variation
end

function call(element::Element, field_name::ASCIIString, xi::VecOrIP, time::Number, variation=nothing)
    field = element(field_name, time, variation)
#   field = isa(variation, Void) ? element[field_name](time) : variation
    basis = get_basis(element)
    return basis(field, xi)
end

function call(element::Element, field_name::ASCIIString, xi::VecOrIP, time::Number, ::Type{Val{:grad}}, variation=nothing)
#   field = isa(variation, Void) ? element[field_name](time) : variation
    field = element(field_name, time, variation)
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


""" Return the jacobian of element. """
function get_jacobian{E}(element::Element{E}, xi::Vector{Float64}, time::Real)
    X = element("geometry", time)
    dN = get_dbasis(E, xi)
    J = sum([kron(dN[:,i], X[i]') for i=1:length(X)])
    return J
end
function get_jacobian{E}(element::Element{E}, ip::IntegrationPoint, time::Real)
    return get_jacobian(element, ip.xi, time)
end

""" Return the determinant of jacobian. """
function LinAlg.det{E<:AbstractElement}(element::Element{E}, xi::Vector{Float64}, time::Real)
    J = get_jacobian(element, xi, time)
    n, m = size(J)
    if n == m
        warn("det(element, ip, time) is ambiguous: use J = get_jacobian(element, ip, time); det(J) instead.")
        return det(J)
    end
    JT = transpose(J)
    if size(JT, 2) == 1
        warn("det(element, ip, time) is ambiguous: use J = get_jacobian(element, ip, time); norm(J) instead.")
        return norm(JT)
    else
        warn("det(element, ip, time) is ambiguous: use J = get_jacobian(element, ip, time); norm(cross(...)) instead.")
        return norm(cross(JT[:,1], JT[:,2]))
    end
end
function LinAlg.det{E<:AbstractElement}(element::Element{E}, ip::IntegrationPoint, time::Real)
    return det(element, ip.xi, time)
end



""" Check does field exist. """
function Base.haskey(element::Element, what)
    haskey(element.fields, what)
end

""" Calculate local normal-tangential coordinates for element. """
function calculate_normal_tangential_coordinates!{E}(element::Element{E}, time::Real)
    ntcoords = Matrix[]
    refcoords = get_reference_element_coordinates(E)
    x = element("geometry", time)
    for xi in refcoords
        dN = get_dbasis(E, xi)*x
        n, m = size(dN)
        @assert n != m # if n == m -> this is not manifold
        if m == 1 # plane case
            tangent = dN / norm(dN)
            normal = [-tangent[2] tangent[1]]'
            push!(ntcoords, [normal tangent])
        elseif m == 2
            normal = cross(dN[:,1], dN[:,2])
            normal /= norm(normal)
            u1 = normal
            j = indmax(abs(u1))
            v2 = zeros(3)
            v2[mod(j,3)+1] = 1.0
            u2 = v2 - dot(u1, v2) / dot(v2, v2) * v2
            u3 = cross(u1, u2)
            tangent1 = u2/norm(u2)
            tangent2 = u3/norm(u3)
            push!(ntcoords, [normal tangent1 tangent2])
        else
            error("calculate_normal_tangential_coordinates!(): n=$n, m=$m")
        end
    end
    element["normal-tangential coordinates"] = ntcoords
end
function calculate_normal_tangential_coordinates!{E}(elements::Vector{Element{E}}, time::Real)
    for element in elements
        calculate_normal_tangential_coordinates!(element, time)
    end
end

""" Pick values from nodes and set to element according to connectivity. """
function update!(element::Element, field_name::ASCIIString, data::Union{Vector, Dict})
    element[field_name] = [data[i] for i in get_connectivity(element)]
end
function update!(elements::Vector{Element}, field_name::ASCIIString, data::Union{Vector, Dict})
#   info("update $field_name for $(length(elements)) elements.")
    for element in elements
        update!(element, field_name, data)
    end
end
