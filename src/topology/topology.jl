# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
    AbstractTopology

Abstract base type for all reference element topologies.

**IMPORTANT SEPARATION OF CONCERNS:**
- **Topology** = Geometric shape (e.g., Triangle, Quadrilateral, Tetrahedron)
- **Basis** = Interpolation scheme (e.g., Lagrange{Triangle, 1}, Serendipity{Quadrilateral, 2})
- **Node count** comes from BASIS, not topology!

A topology defines the **combinatorial structure** of how corner nodes connect to form 
an element in parametric (reference) coordinates. Topologies are **mathematical shapes** 
independent of interpolation schemes or integration rules.

# Key Properties
- Spatial dimension (1D, 2D, 3D)
- Number of **corner** nodes
- Reference element geometry (corner positions only)
- Edge and face connectivity (corner nodes only)
- Node ordering convention

# Topology vs Node Count

The same topology supports different node counts via different basis functions:

```julia
# Same topology (Quadrilateral), different node counts:
Lagrange{Quadrilateral, 1}      → 4 nodes (bilinear)
Serendipity{Quadrilateral, 2}   → 8 nodes (no center)
Lagrange{Quadrilateral, 2}      → 9 nodes (with center)
```

# Examples
```julia
# New API (explicit separation)
topology = Triangle()
basis = Lagrange{Triangle, 1}()
element = Element(basis, (1,2,3))

# Old API (deprecated, but still works via aliases)
element = Element(Tri3, (1,2,3))  # Tri3 is alias for Triangle
```

See also: [`Segment`](@ref), [`Triangle`](@ref), [`Quadrilateral`](@ref), 
          [`Tetrahedron`](@ref), [`Hexahedron`](@ref), [`Pyramid`](@ref), [`Wedge`](@ref)

# Type Parameter

`AbstractTopology{N}` where `N` is the number of nodes. Node count comes from mesh connectivity.

# Examples
```julia
Hexahedron{8} <: AbstractTopology{8}   # 8-node hex (linear)
Hexahedron{20} <: AbstractTopology{20} # 20-node hex (quadratic serendipity)
Hexahedron{27} <: AbstractTopology{27} # 27-node hex (quadratic full)
```

# Rationale

Node count is included in the type parameter for compile-time performance optimization:
- Enables `Val(N)` for zero-allocation ntuple operations
- Allows loop unrolling for small N
- Node count comes from mesh connectivity, not basis choice
- See ADR-002 for detailed design rationale
"""
abstract type AbstractTopology{N} end

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
