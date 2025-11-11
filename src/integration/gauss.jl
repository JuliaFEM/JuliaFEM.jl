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

# ============================================================================
# 1D SEGMENT RULES (Seg2, Seg3)
# ============================================================================
# Generated tensor product: GLSEG1, GLSEG2, GLSEG3, GLSEG4, GLSEG5, ...

get_rule_name(::Gauss{1}, ::Seg2) = :GLSEG1
get_rule_name(::Gauss{2}, ::Seg2) = :GLSEG2
get_rule_name(::Gauss{3}, ::Seg2) = :GLSEG3
get_rule_name(::Gauss{4}, ::Seg2) = :GLSEG4
get_rule_name(::Gauss{5}, ::Seg2) = :GLSEG5

# Seg3 (quadratic) uses same quadrature rules
get_rule_name(::Gauss{1}, ::Seg3) = :GLSEG1
get_rule_name(::Gauss{2}, ::Seg3) = :GLSEG2
get_rule_name(::Gauss{3}, ::Seg3) = :GLSEG3
get_rule_name(::Gauss{4}, ::Seg3) = :GLSEG4
get_rule_name(::Gauss{5}, ::Seg3) = :GLSEG5

# ============================================================================
# 2D TRIANGULAR RULES (Tri3, Tri6, Tri7)
# ============================================================================
# Available: GLTRI1, GLTRI3, GLTRI3B, GLTRI4, GLTRI4B, GLTRI6, GLTRI7, GLTRI12

get_rule_name(::Gauss{1}, ::Tri3) = :GLTRI1
get_rule_name(::Gauss{3}, ::Tri3) = :GLTRI3
get_rule_name(::Gauss{4}, ::Tri3) = :GLTRI4
get_rule_name(::Gauss{6}, ::Tri3) = :GLTRI6
get_rule_name(::Gauss{7}, ::Tri3) = :GLTRI7
get_rule_name(::Gauss{12}, ::Tri3) = :GLTRI12

# Tri6 (quadratic) - needs higher order rules
get_rule_name(::Gauss{1}, ::Tri6) = :GLTRI1
get_rule_name(::Gauss{3}, ::Tri6) = :GLTRI3
get_rule_name(::Gauss{4}, ::Tri6) = :GLTRI4
get_rule_name(::Gauss{6}, ::Tri6) = :GLTRI6
get_rule_name(::Gauss{7}, ::Tri6) = :GLTRI7
get_rule_name(::Gauss{12}, ::Tri6) = :GLTRI12

# Tri7 (quadratic with center) - needs higher order rules
get_rule_name(::Gauss{1}, ::Tri7) = :GLTRI1
get_rule_name(::Gauss{3}, ::Tri7) = :GLTRI3
get_rule_name(::Gauss{4}, ::Tri7) = :GLTRI4
get_rule_name(::Gauss{6}, ::Tri7) = :GLTRI6
get_rule_name(::Gauss{7}, ::Tri7) = :GLTRI7
get_rule_name(::Gauss{12}, ::Tri7) = :GLTRI12

# ============================================================================
# 2D QUADRILATERAL RULES (Quad4, Quad8, Quad9)
# ============================================================================
# Generated tensor product: GLQUAD1, GLQUAD4, GLQUAD9, GLQUAD16, GLQUAD25, ...

get_rule_name(::Gauss{1}, ::Quad4) = :GLQUAD1
get_rule_name(::Gauss{2}, ::Quad4) = :GLQUAD4
get_rule_name(::Gauss{3}, ::Quad4) = :GLQUAD9
get_rule_name(::Gauss{4}, ::Quad4) = :GLQUAD16
get_rule_name(::Gauss{5}, ::Quad4) = :GLQUAD25

# Quad8 (Serendipity) - needs higher order
get_rule_name(::Gauss{1}, ::Quad8) = :GLQUAD1
get_rule_name(::Gauss{2}, ::Quad8) = :GLQUAD4
get_rule_name(::Gauss{3}, ::Quad8) = :GLQUAD9
get_rule_name(::Gauss{4}, ::Quad8) = :GLQUAD16
get_rule_name(::Gauss{5}, ::Quad8) = :GLQUAD25

# Quad9 (quadratic with center) - needs higher order
get_rule_name(::Gauss{1}, ::Quad9) = :GLQUAD1
get_rule_name(::Gauss{2}, ::Quad9) = :GLQUAD4
get_rule_name(::Gauss{3}, ::Quad9) = :GLQUAD9
get_rule_name(::Gauss{4}, ::Quad9) = :GLQUAD16
get_rule_name(::Gauss{5}, ::Quad9) = :GLQUAD25

# ============================================================================
# 3D TETRAHEDRAL RULES (Tet4, Tet10)
# ============================================================================
# Available: GLTET1, GLTET4, GLTET5, GLTET15

get_rule_name(::Gauss{1}, ::Tet4) = :GLTET1
get_rule_name(::Gauss{4}, ::Tet4) = :GLTET4
get_rule_name(::Gauss{5}, ::Tet4) = :GLTET5
get_rule_name(::Gauss{15}, ::Tet4) = :GLTET15

# Tet10 (quadratic) - needs higher order rules
get_rule_name(::Gauss{1}, ::Tet10) = :GLTET1
get_rule_name(::Gauss{4}, ::Tet10) = :GLTET4
get_rule_name(::Gauss{5}, ::Tet10) = :GLTET5
get_rule_name(::Gauss{15}, ::Tet10) = :GLTET15

# ============================================================================
# 3D HEXAHEDRAL RULES (Hex8, Hex20, Hex27)
# ============================================================================
# Generated tensor product: GLHEX1, GLHEX8, GLHEX27, GLHEX64, GLHEX125, ...

get_rule_name(::Gauss{1}, ::Hex8) = :GLHEX1
get_rule_name(::Gauss{2}, ::Hex8) = :GLHEX8
get_rule_name(::Gauss{3}, ::Hex8) = :GLHEX27
get_rule_name(::Gauss{4}, ::Hex8) = :GLHEX64
get_rule_name(::Gauss{5}, ::Hex8) = :GLHEX125

# Hex20 (Serendipity) - needs higher order
get_rule_name(::Gauss{1}, ::Hex20) = :GLHEX1
get_rule_name(::Gauss{2}, ::Hex20) = :GLHEX8
get_rule_name(::Gauss{3}, ::Hex20) = :GLHEX27
get_rule_name(::Gauss{4}, ::Hex20) = :GLHEX64
get_rule_name(::Gauss{5}, ::Hex20) = :GLHEX125

# Hex27 (quadratic with face/volume nodes) - needs higher order
get_rule_name(::Gauss{1}, ::Hex27) = :GLHEX1
get_rule_name(::Gauss{2}, ::Hex27) = :GLHEX8
get_rule_name(::Gauss{3}, ::Hex27) = :GLHEX27
get_rule_name(::Gauss{4}, ::Hex27) = :GLHEX64
get_rule_name(::Gauss{5}, ::Hex27) = :GLHEX125

# ============================================================================
# 3D WEDGE/PRISM RULES (Wedge6, Wedge15)
# ============================================================================
# Available: GLWED6, GLWED6B, GLWED21

get_rule_name(::Gauss{6}, ::Wedge6) = :GLWED6
get_rule_name(::Gauss{21}, ::Wedge6) = :GLWED21

# Wedge15 (quadratic) - needs higher order rules
get_rule_name(::Gauss{6}, ::Wedge15) = :GLWED6
get_rule_name(::Gauss{21}, ::Wedge15) = :GLWED21

# ============================================================================
# 3D PYRAMID RULES (Pyr5)
# ============================================================================
# Available: GLPYR5, GLPYR5B

get_rule_name(::Gauss{5}, ::Pyr5) = :GLPYR5

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

    # Convert to tuple of IntegrationPoints
    # Collect first since quad_data is a zip iterator (cannot be indexed)
    data_vec = collect(quad_data)
    result = ntuple(length(data_vec)) do i
        weight, point = data_vec[i]
        IntegrationPoint(point, weight)  # Type inference from arguments
    end
    return result
end

# Number of integration points
npoints(scheme::Gauss{N}, topology::T) where {N,T<:AbstractTopology} =
    length(integration_points(scheme, topology))

