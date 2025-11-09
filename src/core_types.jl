# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/FEMBase.jl/blob/master/LICENSE

const Node = Vector{Float64}

abstract type AbstractPoint end

# Immutable Point - Dict is a reference type so this is safe
struct Point{P<:AbstractPoint}
    id::UInt
    weight::Float64
    coords::Tuple{Vararg{Float64}}
    fields::Dict{String,AbstractField}  # Reference type, can still be modified
    properties::P
end

function setindex!(point::Point, val::Pair{Float64,T}, field_name) where T
    point.fields[field_name] = field(val)
end

function getindex(point::Point, field_name)
    return point.fields[field_name]
end

function getindex(point::Point, idx::Int)
    return point.coords[idx]
end

function haskey(point::Point, field_name)
    return haskey(point.fields, field_name)
end

function (point::Point)(field_name, time)
    interpolate(point.fields[field_name], time)
end

function Base.iterate(point::Point)
    return Base.iterate(point.coords)
end

function Base.iterate(point::Point, i::Int)
    return Base.iterate(point.coords, i)
end

function update!(point::Point, field_name, val::Pair{Float64,T}) where T
    if haskey(point, field_name)
        update!(point[field_name], val)
    else
        point[field_name] = val
    end
end

#= TODO: in future
type Node <: AbstractPoint
end

type MaterialPoint <: AbstractPoint
end
=#

struct IntegrationPoint <: AbstractPoint
end

const IP = Point{IntegrationPoint}

function IP(id, weight, coords::Tuple)
    return IP(id, weight, coords, Dict(), IntegrationPoint())
end

function IP(id, weight, coords::Vector)
    @warn "Consider giving coordinates as tuple."
    return IP(id, weight, tuple(coords...), Dict(), IntegrationPoint())
end
