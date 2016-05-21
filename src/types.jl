# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

typealias Node Vector{Float64}


"""
Integration point

xi
    (dimensionless) coordinates of integration point
weight
    integration weight
fields
    FieldSet what can be used to store internal variables, stress, strain, ...
"""
immutable IntegrationPoint
    xi :: Vector
    weight :: Float64
    fields :: Dict{ASCIIString, Field}
    changed :: Bool
end

function IntegrationPoint(xi, weight)
    return IntegrationPoint(xi, weight, FieldSet(), false)
end

function setindex!{T<:ForwardDiff.ForwardDiffNumber}(ip::IntegrationPoint, data::Array{T,2}, field_name::ASCIIString)
    data = ForwardDiff.get_value(data)
    setindex!(ip, data, field_name)
end
function setindex!(ip::IntegrationPoint, data, field_name)
    ip.fields[field_name] = Field(data)
    ip.changed = true
end

function getindex(ip::IntegrationPoint, field_name::ASCIIString)
    ip.fields[field_name]
end

function convert(::Type{Number}, ip::IntegrationPoint)
    return ip.xi
end

function call(field::CVTI, ip::IntegrationPoint)
    return call(field, ip.xi)
end

function call(basis::CVTI, field::DCTI, ip::IntegrationPoint)
    call(basis, field, ip.xi)
end

function call(basis::CVTI, field::DVTI, ip::IntegrationPoint, ::Type{Val{:grad}})
    call(basis, field, ip.xi, Val{:grad})
end

function call(basis::CVTI, field::DVTI, ip::IntegrationPoint)
    call(basis, field, ip.xi)
end

function call(basis::CVTI, geometry::DVTI, field::Union{DCTI, DVTI}, ip::IntegrationPoint, ::Type{Val{:grad}})
    call(basis, geometry, field, ip.xi, Val{:grad})
end

#function Base.call(basis::Basis, increment::Increment, ip::IntegrationPoint)
#    return call(basis, increment, ip.xi)
#end
#function Base.call(basis::Basis, increment::Increment, ip::IntegrationPoint, ::Type{Val{:grad}})
#    return call(basis, increment, ip.xi, Val{:grad})
#end
#function Base.call(basis::Basis, field::Field, ip::IntegrationPoint, ::Type{Val{:grad}})
#    return call(basis, field, ip.xi, Val{:grad})
#end
#function Base.call(basis::Basis, geometry::Increment, field::Increment, ip::IntegrationPoint, ::Type{Val{:grad}})
#    return call(basis, geometry, field, ip.xi, Val{:grad})
#end
#function Base.call(basis::Basis, field::Field, ip::IntegrationPoint)
#    return call(basis, field, ip.xi)
#end
