# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
    AbstractTopology

Abstract base type for all reference element topologies.

A topology defines the combinatorial structure of how nodes connect to form an element
in parametric (reference) coordinates. Topologies are mathematical objects independent
of interpolation schemes or integration rules.

# Key Properties
- Number of nodes
- Spatial dimension (1D, 2D, 3D)
- Reference element geometry
- Node ordering convention

# Examples
```julia
Tri3()   # 3-node triangle
Quad4()  # 4-node quadrilateral
Tet10()  # 10-node tetrahedron
Hex8()   # 8-node hexahedron
```

See also: [`Tri3`](@ref), [`Quad4`](@ref), [`Tet10`](@ref), [`Hex8`](@ref)
"""
abstract type AbstractTopology end

"""
    nnodes(topology::AbstractTopology) -> Int

Return the number of nodes in the reference element.

# Examples
```julia
julia> nnodes(Tri3())
3

julia> nnodes(Hex8())
8
```
"""
function nnodes end

"""
    dim(topology::AbstractTopology) -> Int

Return the spatial dimension of the reference element (1, 2, or 3).

# Examples
```julia
julia> dim(Tri3())
2

julia> dim(Hex8())
3
```
"""
function dim end

"""
    reference_coordinates(topology::AbstractTopology) -> NTuple{N, NTuple{D, Float64}}

Return the coordinates of nodes in the reference element as a tuple of tuples.

**Zero allocation:** Returns compile-time sized tuple, fully stack allocated.

# Convention
Reference elements are defined in parametric coordinates ξ ∈ [-1, 1]^D (for most elements).

# Examples
```julia
julia> reference_coordinates(Tri3())
((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))

julia> typeof(reference_coordinates(Tri3()))
NTuple{3, NTuple{2, Float64}}
```
"""
function reference_coordinates end

"""
    faces(topology::AbstractTopology) -> NTuple{Nf, NTuple{Nn, Int}}

Return the connectivity of faces for the reference element as a tuple of tuples.

Each face is represented as a tuple of local node indices (1-based).

**Zero allocation:** Returns compile-time sized nested tuple, fully stack allocated.

# Examples
```julia
julia> faces(Quad4())
((1, 2, 3, 4),)  # 2D element has one face (itself)

julia> faces(Hex8())
((1, 4, 3, 2), (5, 6, 7, 8), (1, 2, 6, 5), (2, 3, 7, 6), (3, 4, 8, 7), (4, 1, 5, 8))
```
"""
function faces end

"""
    edges(topology::AbstractTopology) -> NTuple{Ne, Tuple{Int, Int}}

Return the connectivity of edges for the reference element as a tuple of tuples.

Each edge is represented as a tuple of two local node indices (1-based).

**Zero allocation:** Returns compile-time sized tuple, fully stack allocated.

# Examples
```julia
julia> edges(Tri3())
((1, 2), (2, 3), (3, 1))

julia> typeof(edges(Tri3()))
NTuple{3, Tuple{Int64, Int64}}
```
"""
function edges end
