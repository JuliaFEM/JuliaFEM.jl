# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

### INTEGRATIONPOINT ###

"""
Integration point

xi :: Array{Float64, 1}
    (dimensionless) coordinates of integration point
weight :: Float64
    Integration weight
attributes :: Dict{Any, Any}
    This is used to save internal variables of IP needed e.g. for incremental
    material models.
"""
type IntegrationPoint
    xi :: Vector
    weight :: Float64
    fields :: Dict{ASCIIString, FieldSet}
end
function IntegrationPoint(xi, weight)
    IntegrationPoint(xi, weight, Dict())
end

call(N::Basis, ip::IntegrationPoint) = N(ip.xi)


