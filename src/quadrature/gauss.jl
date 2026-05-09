# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Gauss-Legendre integration using new quadrature API.
"""

"""
    integration_points(topology::AbstractTopology)
    -> Tuple{Vararg{QuadraturePoint{D}}}

Return integration points for the given topology using default quadrature rule.
"""
function integration_points(topology::T) where {T<:AbstractTopology}
    # Default quadrature rule for this topology (basis order inferred
    # from node count via `_infer_basis_order`).
    rule = default_quadrature(T)

    # Map parametric topology type to generic quadrature type
    # (e.g. Hexahedron{8} → Hexahedron, Triangle{3} → Triangle).
    quad_topo = _quadrature_topology_type(T)

    # Returns SVector{N, QuadraturePoint{D,Float64}}. Wrapping in a
    # `Tuple` keeps the return type `NTuple{N, QuadraturePoint{...}}`
    # so call sites can still splat / destructure; for the small N
    # used here the conversion compiles to a stack-only construction.
    quad_points = get_quadrature_points(quad_topo, rule)
    return Tuple(quad_points)
end

"""
    npoints(topology::AbstractTopology) -> Int

Return the number of integration points for the given topology.
"""
npoints(topology::AbstractTopology) = length(integration_points(topology))

