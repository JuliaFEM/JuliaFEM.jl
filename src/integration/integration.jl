# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
    AbstractIntegration

Abstract base type for all numerical integration (quadrature) schemes.

An integration scheme defines how to numerically integrate over a reference element
by specifying integration point locations and weights. Integration schemes are
independent of element topology and interpolation schemes (though the number of
points needed may depend on polynomial order).

# Key Properties
- Integration points (locations in parametric space)
- Weights
- Accuracy order

# Examples
```julia
Gauss{2}()    # 2-point Gauss quadrature
Gauss{3}()    # 3-point Gauss quadrature
Lobatto{3}()  # 3-point Gauss-Lobatto quadrature
Reduced()     # Reduced integration (element-dependent)
```

See also: [`Gauss`](@ref), [`Lobatto`](@ref), [`IntegrationPoint`](@ref)
"""
abstract type AbstractIntegration end

"""
    IntegrationPoint{D}

Represents a single integration point in D-dimensional parametric space.

# Fields
- `ξ::NTuple{D, Float64}`: Location in parametric coordinates
- `weight::Float64`: Integration weight

# Examples
```julia
ip = IntegrationPoint((0.0, 0.0), 1.0)  # 2D point at origin with weight 1
```
"""
struct IntegrationPoint{D}
    ξ::NTuple{D,Float64}
    weight::Float64
end

"""
    integration_points(scheme::AbstractIntegration, topology::AbstractTopology)
    -> NTuple{N, IntegrationPoint{D}}

Return the integration points and weights for the given integration scheme
applied to the reference element topology.

**Zero allocation:** Returns compile-time sized tuple of IntegrationPoints for
known quadrature rules. Falls back to Vector for dynamic rules.

# Arguments
- `scheme`: Integration scheme (e.g., `Gauss{3}()`)
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
function integration_points end

"""
    npoints(scheme::AbstractIntegration, topology::AbstractTopology) -> Int

Return the number of integration points for the given scheme and topology.

# Examples
```julia
julia> npoints(Gauss{2}(), Tri3())
3

julia> npoints(Gauss{2}(), Quad4())
4
```
"""
function npoints end

"""
    default_integration(topology::Type{<:AbstractTopology{N}}) where N
    -> AbstractIntegration

Return the default (recommended) integration scheme for a given topology type.

# Default Rules
- Linear elements (P1): Use minimal integration that's exact for linear basis
- Quadratic elements (P2): Use integration exact for quadratic basis

# Examples
```julia
julia> default_integration(Hexahedron{8})
Gauss{2}()  # 2×2×2 = 8 points (exact for trilinear)

julia> default_integration(Tetrahedron{4})
Gauss{1}()  # 1 point (exact for linear)

julia> default_integration(Hexahedron{27})
Gauss{3}()  # 3×3×3 = 27 points (exact for triquadratic)
```
"""
function default_integration end

# Default integration rules for common topologies
default_integration(::Type{Tetrahedron{4}}) = Gauss{1}()
default_integration(::Type{Tetrahedron{10}}) = Gauss{2}()
default_integration(::Type{Hexahedron{8}}) = Gauss{2}()
default_integration(::Type{Hexahedron{20}}) = Gauss{3}()
default_integration(::Type{Hexahedron{27}}) = Gauss{3}()
default_integration(::Type{Triangle{3}}) = Gauss{1}()
default_integration(::Type{Triangle{6}}) = Gauss{2}()
default_integration(::Type{Quadrilateral{4}}) = Gauss{2}()
default_integration(::Type{Quadrilateral{8}}) = Gauss{3}()
default_integration(::Type{Quadrilateral{9}}) = Gauss{3}()
