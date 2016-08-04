# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

abstract AbstractElement

type Element{E<:AbstractElement}
    id :: Int
    connectivity :: Vector{Int}
    integration_points :: Vector{IP}
    fields :: Dict{AbstractString, Field}
    properties :: E
end

function Element{E<:AbstractElement}(::Type{E}, id::Int64, connectivity::Vector{Int64})
    return Element{E}(id, connectivity, [], Dict(), E())
end

function Element{E<:AbstractElement}(::Type{E}, connectivity::Vector{Int64})
    return Element{E}(-1, connectivity, [], Dict(), E())
end

function getindex(element::Element, field_name::AbstractString)
    return element.fields[field_name]
end

function setindex!(element::Element, data::Field, field_name)
    element.fields[field_name] = data
end

function setindex!(element::Element, data::Function, field_name)
    if method_exists(data, Tuple{Element, Vector, Float64})
        # create enclosure to pass element as argument
        function wrapper_(ip, time)
            return data(element, ip, time)
        end
        field = Field(wrapper_)
    else
        field = Field(data)
    end
    element.fields[field_name] = field
end

function setindex!(element::Element, data, field_name)
    element.fields[field_name] = Field(data)
end

function call(element::Element, field_name)
    return element[field_name]
end

function call(element::Element, field_name, time)
    return element[field_name](time)
end

function last(element::Element, field_name::AbstractString)
    return last(element[field_name])
end

function call(element::Element, ip, time::Float64=0.0)
    return get_basis(element, ip, time)
end

function call(element::Element, ip, time::Float64, ::Type{Val{:Jacobian}})
    X = element("geometry", time)
    dN = get_dbasis(element, ip, time)
    nbasis = length(element)
    if isa(X.data, Vector)
        J = sum([kron(dN[:,i], X[i]') for i=1:nbasis])
    else
        c = get_connectivity(element)
        J = sum([kron(dN[:,i], X[c[i]]') for i=1:nbasis])
    end
    return J
end

function call(element::Element, ip, time::Float64, ::Type{Val{:detJ}})
    J = element(ip, time, Val{:Jacobian})
    n, m = size(J)
    if n == m  # volume element
        return det(J)
    end
    JT = transpose(J)
    if size(JT, 2) == 1  # boundary of 2d problem, || ∂X/∂ξ ||
        return norm(JT)
    else # manifold on 3d problem, || ∂X/∂ξ₁ × ∂X/∂ξ₂ ||
        return norm(cross(JT[:,1], JT[:,2]))
    end
end

function call(element::Element, ip, time::Float64, ::Type{Val{:Grad}})
    J = element(ip, time, Val{:Jacobian})
    return inv(J)*get_dbasis(element, ip, time)
end

function call(element::Element, field_name::AbstractString, ip, time::Float64, ::Type{Val{:Grad}})
    return element(ip, time, Val{:Grad})*element[field_name](time)
end

function call(element::Element, field::Field, time)
    return field(time)
end

function call(element::Element, field::DCTI, time)
    return field.data
end

function call(element::Element, field_name::AbstractString, time)
    field = element[field_name]
    return element(field, time)
end

function call(element::Element, field_name::AbstractString, ip, time::Float64)
    field = element[field_name]
    return element(field, ip, time)
end

function call(element::Element, field::DCTI, ip, time::Float64)
    return field.data
end

function call(element::Element, field::DCTV, ip, time::Float64)
    return field(time).data
end

function call(element::Element, field::CVTV, ip, time::Float64)
    return field(ip, time)
end

function call(element::Element, field::Field, ip, time::Float64)
    field_ = field(time)
    basis = element(ip, time)
    n = length(element)
    if isa(field_.data, Vector)
        m = length(field_)
        if n != m
            error("Error when trying to interpolate field $field at coords $ip and time $time: element length is $n and field length is $m, f = Nᵢfᵢ makes no sense!")
        end
        return sum([field_[i]*basis[i] for i=1:n])
    else
        c = get_connectivity(element)
        return sum([field_[c[i]]*basis[i] for i=1:n])
    end
end

function size(element::Element, dim)
    return size(element)[dim]
end

""" Update element field based on a dictionary of nodal data and connectivity information.

Examples
--------
julia> data = Dict(1 => [0.0, 0.0], 2 => [1.0, 2.0])
julia> element = Seg2([1, 2])
julia> update!(element, "geometry", data)

As a result element now have time invariant (variable) vector field "geometry" with data ([0.0, 0.0], [1.0, 2.0]).

"""
function update!(element::Element, field_name, data::Dict)
    element[field_name] = Field(data)
    #element[field_name] = [data[i] for i in get_connectivity(element)]
end

function update!{K,V}(element::Element, field_name, data::Pair{Float64, Dict{K, V}})
    #time, field_data = data
    #element_data = V[field_data[i] for i in get_connectivity(element)]
    #update!(element, field_name, time => element_data)
    if haskey(element, field_name)
        update!(element[field_name], data)
    else
        element[field_name] = Field(data)
    end
end

function update!(element::Element, field_name::AbstractString, datas::Union{Real, Vector, Pair{Float64, Union{Float64, Real, Vector{Any}}}}...)
    for data in datas
        if haskey(element, field_name)
            update!(element[field_name], data)
        else
            if length(data) != length(element)
                update!(element, field_name, DCTI(data))
            else
                element[field_name] = data
            end
        end
    end
end

#=
function update!(element::Element, field_name, data::Pair...)
    for data in datas
        update!(element, field_name, data)
    end
end
=#

function update!(element::Element, field_name, data::Pair{Float64, Vector{Any}})
    if haskey(element, field_name)
        update!(element[field_name], data)
    else
        element[field_name] = data
    end
end

function update!(element::Element, field_name, data::Pair{Float64, Vector{Int64}})
    if haskey(element, field_name)
        update!(element[field_name], data)
    else
        element[field_name] = data
    end
end

function update!(element::Element, field_name, data::Pair{Float64, Vector{Float64}})
    if haskey(element, field_name)
        update!(element[field_name], data)
    else
        element[field_name] = data
    end
end

function update!(element::Element, field_name, data::Pair{Float64, Vector{Vector{Float64}}})
    if haskey(element, field_name)
        update!(element[field_name], data)
    else
        element[field_name] = data
    end
end

function update!(element::Element, field_name, data::Pair{Float64, Float64})
    if haskey(element, field_name)
        update!(element[field_name], data)
    else
        element[field_name] = data
    end
end

function update!(element::Element, field_name::AbstractString, data::Union{Float64, Vector})
    if haskey(element, field_name)
        update!(element[field_name], data)
    else
        if length(data) != length(element)
            update!(element, field_name, DCTI(data))
        else
            element[field_name] = data
        end
    end
end

function update!(element::Element, datas::Pair...)
    for (field_name, data) in datas
        if haskey(element, field_name)
            update!(element[field_name], data)
        else
            element[field_name] = data
        end
    end
end

function update!(element::Element, field_name, data::Function)
    element[field_name] = data
end

function update!(element::Element, field_name, field::Field)
    element[field_name] = field
end

function update!(elements::Vector, field_name, data)
    for element in elements
        update!(element, field_name, data)
    end
end

""" Check existence of field. """
function haskey(element::Element, field_name)
    haskey(element.fields, field_name)
end

function get_connectivity(element::Element)
    return element.connectivity
end

function get_integration_points(element::Element)
    # first time initialize default integration points
    if length(element.integration_points) == 0
        ips = get_integration_points(element.properties)
        element.integration_points = [IP(i, w, xi) for (i, (w, xi)) in enumerate(ips)]
    end
    return element.integration_points
end

""" This is a special case, temporarily change order
of integration scheme mainly for mass matrix.
"""
function get_integration_points(element::Element, change_order::Int)
    order = get_integration_order(element.properties)
    order += change_order
    ips = get_integration_points(element.properties, order)
    return [IP(i, w, xi) for (i, (w, xi)) in enumerate(ips)]
end

function get_gdofs(element::Element)
    return get_gdofs(element, 1)
end

""" Return dual basis transformation matrix Ae. """
function get_dualbasis(element::Element, time::Float64, order=1)
    nnodes = length(element)
    De = zeros(nnodes, nnodes)
    Me = zeros(nnodes, nnodes)
    for ip in get_integration_points(element, order)
        detJ = element(ip, time, Val{:detJ})
        w = ip.weight*detJ
        N = element(ip, time)
        De += w*diagm(vec(N))
        Me += w*N'*N
    end
    return De, Me, De*inv(Me)
end

""" Find inverse isoparametric mapping of element. """
function get_local_coordinates(element::Element, X::Vector, time::Float64; max_iterations=10, tolerance=1.0e-6)
    haskey(element, "geometry") || error("element geometry not defined, cannot calculate inverse isoparametric mapping")
    dim = size(element, 1)
    dim == length(X) || error("manifolds not supported.")
    xi = zeros(dim)
    dX = element("geometry", xi, time) - X
    for i=1:max_iterations
        J = element(xi, time, Val{:Jacobian})'
        xi -= J \ dX
        dX = element("geometry", xi, time) - X
        norm(dX) < tolerance && return xi
    end
    info("X = $X, dX = $dX, xi = $xi")
    error("Unable to find inverse isoparametric mapping for element $element for X = $X")
end

""" Test is X inside element. """
function inside{E}(element::Element{E}, X, time)
    xi = get_local_coordinates(element, X, time)
    return inside(E, xi)
end
