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
    # Get default quadrature rule for this topology (uses node count to infer order)
    rule = default_quadrature(T)
    
    # Map parametric topology type to generic quadrature type
    # E.g., Hexahedron{8} → Hexahedron, Triangle{3} → Triangle
    quad_topo = _quadrature_topology_type(T)
    
    # Get points using new API: get_quadrature_points(Hexahedron, GaussLegendre{2}())
    # Returns: SVector{N, QuadraturePoint{D,Float64}}
    quad_points = get_quadrature_points(quad_topo, rule)
    
    # Convert SVector to Tuple (zero-allocation)
    return Tuple(quad_points)
end

"""
    npoints(topology::AbstractTopology) -> Int

Return the number of integration points for the given topology.
"""
npoints(topology::AbstractTopology) = length(integration_points(topology))

