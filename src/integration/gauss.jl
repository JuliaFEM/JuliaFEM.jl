# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
    Gauss{N} <: AbstractIntegration

Gauss-Legendre quadrature with N points per dimension.

Gauss quadrature is optimal for polynomial integration: N points integrate
polynomials of degree 2N-1 exactly.

# Type Parameter
- `N::Int`: Number of integration points per dimension (or order indicator)

# Implementation Note
This is a thin wrapper around the existing quadrature rules in src/quadrature/
(consolidated from FEMQuad.jl). The actual integration points and weights are
provided by the FEMQuad module.

# Total Points
- 1D line: N points (e.g., :GLSEG2, :GLSEG4)
- 2D quad: N² points (e.g., :GLQUAD4, :GLQUAD9)
- 2D triangle: Variable (e.g., :GLTRI1, :GLTRI3, :GLTRI6, :GLTRI7)
- 3D hex: N³ points (e.g., :GLHEX8, :GLHEX27)
- 3D tetrahedron: Variable (e.g., :GLTET1, :GLTET4)

# Examples
```julia
# 1-point quadrature (degree 1 polynomials)
Gauss{1}()  # Maps to :GLTRI1, :GLTET1, etc.

# 3-point quadrature 
Gauss{3}()  # Maps to :GLTRI3, :GLQUAD9, etc.

# Get integration points for specific topology
ips = integration_points(Gauss{3}(), Tri3())
```

# References
- Abramowitz & Stegun, "Handbook of Mathematical Functions"
- Dunavant, "High degree efficient symmetrical Gaussian quadrature rules for the triangle"

See also: [`AbstractIntegration`](@ref), [`Lobatto`](@ref), [`integration_points`](@ref)
"""
struct Gauss{N} <: AbstractIntegration end

# Note: Integration point data comes from src/quadrature/*.jl
# Functions get_quadrature_points() and get_order() are defined there
# and available in parent module scope (included via src/quadrature.jl)

"""
    get_rule_name(::Gauss{N}, topology::AbstractTopology) -> Symbol

Map Gauss{N} + topology to the corresponding FEMQuad rule name.

# Examples
```julia
julia> get_rule_name(Gauss{1}(), Tri3())
:GLTRI1

julia> get_rule_name(Gauss{3}(), Tri3())
:GLTRI3

julia> get_rule_name(Gauss{2}(), Quad4())
:GLQUAD4
```
"""
function get_rule_name end

# 1D rules (segments)
get_rule_name(::Gauss{1}, ::Type{<:AbstractTopology}) = :GLSEG1
get_rule_name(::Gauss{2}, ::Type{<:AbstractTopology}) = :GLSEG2
get_rule_name(::Gauss{3}, ::Type{<:AbstractTopology}) = :GLSEG3
get_rule_name(::Gauss{4}, ::Type{<:AbstractTopology}) = :GLSEG4
get_rule_name(::Gauss{5}, ::Type{<:AbstractTopology}) = :GLSEG5

# 2D triangular rules
get_rule_name(::Gauss{1}, ::Tri3) = :GLTRI1
get_rule_name(::Gauss{3}, ::Tri3) = :GLTRI3
get_rule_name(::Gauss{4}, ::Tri3) = :GLTRI4
get_rule_name(::Gauss{6}, ::Tri3) = :GLTRI6
get_rule_name(::Gauss{7}, ::Tri3) = :GLTRI7

# 2D quadrilateral rules (tensor product)
get_rule_name(::Gauss{1}, ::Quad4) = :GLQUAD1
get_rule_name(::Gauss{2}, ::Quad4) = :GLQUAD4
get_rule_name(::Gauss{3}, ::Quad4) = :GLQUAD9
get_rule_name(::Gauss{4}, ::Quad4) = :GLQUAD16
get_rule_name(::Gauss{5}, ::Quad4) = :GLQUAD25

# 3D tetrahedral rules
# get_rule_name(::Gauss{1}, ::Tet4) = :GLTET1
# get_rule_name(::Gauss{4}, ::Tet4) = :GLTET4
# get_rule_name(::Gauss{5}, ::Tet4) = :GLTET5
# get_rule_name(::Gauss{15}, ::Tet4) = :GLTET15

# 3D hexahedral rules (tensor product)
# get_rule_name(::Gauss{2}, ::Hex8) = :GLHEX8
# get_rule_name(::Gauss{3}, ::Hex8) = :GLHEX27
# get_rule_name(::Gauss{4}, ::Hex8) = :GLHEX64
# get_rule_name(::Gauss{5}, ::Hex8) = :GLHEX125

# 3D wedge rules (triangular prism)
# get_rule_name(::Gauss{6}, ::Wedge6) = :GLWED6
# get_rule_name(::Gauss{21}, ::Wedge6) = :GLWED21

# 3D pyramid rules
# get_rule_name(::Gauss{5}, ::Pyr5) = :GLPYR5

"""
    integration_points(scheme::Gauss{N}, topology::AbstractTopology)
    -> Tuple{Vararg{IntegrationPoint{D}}}

Return the integration points and weights for Gauss-Legendre quadrature
on the given topology.

**Zero allocation:** Returns tuple of IntegrationPoints (stack allocated).

# Arguments
- `scheme`: Gauss quadrature scheme (e.g., `Gauss{3}()`)
- `topology`: Reference element topology (e.g., `Tri3()`)

# Returns
Tuple of `IntegrationPoint` with locations ξ and weights.

# Examples
```julia
julia> ips = integration_points(Gauss{1}(), Tri3())
(IntegrationPoint{2}((0.333..., 0.333...), 0.5),)

julia> typeof(ips)
Tuple{IntegrationPoint{2}}
```
"""
function integration_points(scheme::Gauss{N}, topology::T) where {N,T<:AbstractTopology}
    rule_name = get_rule_name(scheme, topology)
    D = dim(topology)

    # Get points from quadrature module (src/quadrature/)
    quad_data = get_quadrature_points(Val{rule_name})

    # Convert to tuple of IntegrationPoints (zero allocation)
    return tuple((IntegrationPoint{D}(point, weight) for (weight, point) in quad_data)...)
end

# Number of integration points
npoints(scheme::Gauss{N}, topology::T) where {N,T<:AbstractTopology} =
    length(integration_points(scheme, topology))

