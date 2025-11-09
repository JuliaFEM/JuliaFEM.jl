# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
    Quadrilateral <: AbstractTopology

Quadrilateral element topology in 2D (reference element geometry).

**Important:** This type defines ONLY the geometric shape. Node count is determined
by the interpolation scheme (basis functions):
- `Lagrange{Quadrilateral, 1}` → 4 nodes (Q1, bilinear)
- `Serendipity{Quadrilateral, 2}` → 8 nodes (Q2, no center node)
- `Lagrange{Quadrilateral, 2}` → 9 nodes (Q2, full tensor product)
- `Lagrange{Quadrilateral, 3}` → 16 nodes (Q3, cubic)

# Reference Element
```
     η
     ^
     |
  4  |  3
     +-----+
     |     |
     |  +  | --> ξ
     |     |
     +-----+
  1        2
```

# Standard Node Positions
For `Lagrange{Quadrilateral, 1}` (Q1, 4 nodes):
1. (-1, -1) - Bottom-left
2. ( 1, -1) - Bottom-right
3. ( 1,  1) - Top-right
4. (-1,  1) - Top-left

For higher-order bases, additional nodes are added on edges, faces, and interior.

# Topology Properties
- Dimension: 2
- Edges: 4
- Faces: 1 (the element itself in 2D)

# Typical Usage
```julia
julia> topology = Quadrilateral()
julia> dim(topology)
2
julia> basis = Lagrange{Quadrilateral, 1}()
julia> nnodes(basis)  # Node count comes from BASIS, not topology
4
```

See also: [`AbstractTopology`](@ref), [`Triangle`](@ref), [`Lagrange`](@ref), [`Serendipity`](@ref)
"""
struct Quadrilateral <: AbstractTopology end

dim(::Quadrilateral) = 2

"""
    reference_coordinates(::Quadrilateral)

Get standard reference node positions for Quadrilateral (4 corners in parametric space).
These are the Q1 (bilinear) node positions by default.
"""
function reference_coordinates(::Quadrilateral)
    return (
        (-1.0, -1.0),  # Node 1: Bottom-left
        (1.0, -1.0),  # Node 2: Bottom-right
        (1.0, 1.0),  # Node 3: Top-right
        (-1.0, 1.0),  # Node 4: Top-left
    )
end

function edges(::Quadrilateral)
    return (
        (1, 2),  # Edge 1: Bottom
        (2, 3),  # Edge 2: Right
        (3, 4),  # Edge 3: Top
        (4, 1),  # Edge 4: Left
    )
end

# For 2D elements, faces are the element itself
faces(::Quadrilateral) = ((1, 2, 3, 4),)

# ============================================================================
# Deprecated aliases (for backwards compatibility)
# ============================================================================

"""
    Quad4

**DEPRECATED:** Use `Quadrilateral` + `Lagrange{Quadrilateral, 1}` instead.

The old `Quad4` type conflated topology (quadrilateral) with node count (4).
In the new architecture:
- Topology defines geometric shape only
- Basis functions determine node count

Migration:
```julia
# Old (deprecated)
element = Element(Quad4, (1,2,3,4))

# New (correct)
element = Element(Quadrilateral(), Lagrange{Quadrilateral, 1}(), Gauss{2}(), (1,2,3,4))
```
"""
const Quad4 = Quadrilateral

# Note: Quad4 is deprecated. Use Quadrilateral with parametric Lagrange basis instead.
