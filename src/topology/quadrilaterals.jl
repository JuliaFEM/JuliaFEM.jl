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
  N4 |  N3
     +-----+
     |     |
     |  +  | --> ξ
     |     |
     +-----+
  N1        N2
```

# Standard Corner Node Positions
1. (-1, -1) - Bottom-left
2. ( 1, -1) - Bottom-right
3. ( 1,  1) - Top-right
4. (-1,  1) - Top-left

# Topology Properties
- Dimension: 2
- Corner nodes: 4
- Edges: 4
- Faces: 1 (the element itself in 2D)

# Typical Usage
```julia
julia> topology = Quadrilateral()
julia> dim(topology)
2
julia> reference_coordinates(topology)  # Corner nodes only
((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))

julia> basis = Lagrange{Quadrilateral, 1}()
julia> nnodes(basis)  # Bilinear: 4 nodes
4

julia> basis = Serendipity{Quadrilateral, 2}()
julia> nnodes(basis)  # Serendipity: 8 nodes (no center)
8

julia> basis = Lagrange{Quadrilateral, 2}()
julia> nnodes(basis)  # Full Lagrange: 9 nodes (with center)
9
```

**Zero allocation:** All functions return compile-time sized tuples.

See also: [`AbstractTopology`](@ref), [`Triangle`](@ref), [`Lagrange`](@ref), [`Serendipity`](@ref)
"""
struct Quadrilateral <: AbstractTopology end

dim(::Quadrilateral) = 2

"""
    reference_coordinates(::Quadrilateral)

Get corner node positions for Quadrilateral (4 vertices in parametric space).
"""
function reference_coordinates(::Quadrilateral)
    return (
        (-1.0, -1.0),  # Node 1: Bottom-left
        (1.0, -1.0),  # Node 2: Bottom-right
        (1.0, 1.0),  # Node 3: Top-right
        (-1.0, 1.0),  # Node 4: Top-left
    )
end

"""
    edges(::Quadrilateral)

Edge connectivity for quadrilateral (corner nodes).
"""
function edges(::Quadrilateral)
    return (
        (1, 2),  # Edge 1: Bottom
        (2, 3),  # Edge 2: Right
        (3, 4),  # Edge 3: Top
        (4, 1),  # Edge 4: Left
    )
end

"""
    faces(::Quadrilateral)

For 2D elements, the face is the element itself.
"""
faces(::Quadrilateral) = ((1, 2, 3, 4),)

# ============================================================================
# Deprecated aliases (for backwards compatibility)
# ============================================================================

"""
    Quad4

**DEPRECATED:** Backward compatibility alias. Use `Quadrilateral` with `Lagrange{Quadrilateral, 1}`.

The old `Quad4` conflated topology (quadrilateral) with node count (4).
In the new architecture:
- Topology defines geometric shape only
- Basis functions determine node count

This alias allows old code to work:
```julia
# Old style (still works)
element = Element(Quad4, (1, 2, 3, 4))
# Internally converted to:
element = Element(Quadrilateral, (1, 2, 3, 4))  # Infers Lagrange{Quadrilateral, 1}
```

New code should use explicit topology + basis:
```julia
element = Element(Lagrange{Quadrilateral, 1}, (1, 2, 3, 4))
```
"""
const Quad4 = Quadrilateral

"""
    Quad8

**DEPRECATED:** Backward compatibility alias. Use `Quadrilateral` with `Serendipity{Quadrilateral, 2}`.

8-node quadrilateral with serendipity basis (NO center node).

This alias allows old code to work:
```julia
# Old style (still works)
element = Element(Quad8, (1, 2, 3, 4, 5, 6, 7, 8))
# Internally converted to:
element = Element(Quadrilateral, (1, 2, 3, 4, 5, 6, 7, 8))  # Infers Serendipity
```

New code should be explicit about basis family:
```julia
element = Element(Serendipity{Quadrilateral, 2}, (1, 2, 3, 4, 5, 6, 7, 8))
```
"""
const Quad8 = Quadrilateral  # Same topology! Basis determines node pattern.

"""
    Quad9

**DEPRECATED:** Backward compatibility alias. Use `Quadrilateral` with `Lagrange{Quadrilateral, 2}`.

9-node quadrilateral with full tensor product basis (WITH center node).

This alias allows old code to work:
```julia
# Old style (still works)
element = Element(Quad9, (1, 2, 3, 4, 5, 6, 7, 8, 9))
# Internally converted to:
element = Element(Quadrilateral, (1, 2, 3, 4, 5, 6, 7, 8, 9))  # Infers Lagrange
```

New code should be explicit:
```julia
element = Element(Lagrange{Quadrilateral, 2}, (1, 2, 3, 4, 5, 6, 7, 8, 9))
```
"""
const Quad9 = Quadrilateral  # Same topology! Basis determines node pattern.
