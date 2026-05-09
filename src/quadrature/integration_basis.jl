# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
    basis_quadrature_order(basis::AbstractBasis) -> Int

Polynomial order passed to [`default_quadrature(::Type{<:AbstractTopology}, ::Int)`](@ref)
when building [`integration_points(topology, basis)`](@ref).

Extensible: add methods for new [`AbstractBasis`](@ref) subtypes used in assembly caches.
Patch/tensor-product bases used in IGA (B-spline/NURBS on a parameter cell) should
implement this hook so quadrature matches the quadrature rule degree of the weak form,
analogous to [`Lagrange`](@ref) / [`Serendipity`](@ref).
"""
basis_quadrature_order(::Lagrange{P}) where {P} = P
basis_quadrature_order(::Serendipity{P}) where {P} = P

"""
    integration_points(topology::AbstractTopology, basis::AbstractBasis)

Like [`integration_points(topology)`](@ref), but selects [`default_quadrature`](@ref)
from **`basis`** via [`basis_quadrature_order`](@ref) instead of inferring order only
from [`nnodes`](@ref). Use this when the mesh topology node count does not reflect
the polynomial degree of the active basis (or to force a consistent rule with
[`create_element_cache`](@ref)).
"""
function integration_points(topology::T, basis::AbstractBasis) where {T<:AbstractTopology}
    ord = basis_quadrature_order(basis)
    rule = default_quadrature(T, ord)
    quad_topo = _quadrature_topology_type(T)
    quad_points = get_quadrature_points(quad_topo, rule)
    return Tuple(quad_points)
end
