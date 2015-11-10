# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

function ForwardDiff.derivative{T}(f::Function, S::Matrix{T}, args...)
    shape = size(S)
    wrapper(S::Vector) = f(reshape(S, shape))
    deriv = ForwardDiff.gradient(wrapper, vec(S), args...)
    return reshape(deriv, shape)
end

"""
Integration point

xi
    (dimensionless) coordinates of integration point
weight
    integration weight
fields
    FieldSet what can be used to store internal variables, stress, strain, ...
"""
type IntegrationPoint
    xi :: Vector
    weight :: Float64
    fields :: FieldSet
end

function IntegrationPoint(xi, weight)
    IntegrationPoint(xi, weight, Dict())
end

function Base.convert(::Type{Number}, ip::IntegrationPoint)
    return ip.xi
end

function Base.call(basis::Basis, ip::IntegrationPoint)
    return basis(ip.xi)
end

function Base.call(basis::Basis, increment::Increment, ip::IntegrationPoint)
    return call(basis, increment, ip.xi)
end

function Base.call(basis::Basis, increment::Increment, ip::IntegrationPoint, ::Type{Val{:grad}})
    return call(basis, increment, ip.xi, Val{:grad})
end

function Base.call(basis::Basis, geometry::Increment, field::Increment, ip::IntegrationPoint, ::Type{Val{:grad}})
    return call(basis, geometry, field, ip.xi, Val{:grad})
end
