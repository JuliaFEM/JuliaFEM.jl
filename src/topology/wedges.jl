# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE

"""
    Wedge <: AbstractTopology

Wedge/prism element topology (3D, triangular prism).

**Important:** This type defines ONLY the geometric shape. Node count is determined
by the interpolation scheme (basis functions):
- `Lagrange{Wedge, 1}` → 6 nodes (linear)
- `Lagrange{Wedge, 2}` → 15 nodes (quadratic, with edge midpoints)

# Reference Element
```
      N6
      /|\\
     / | \\
   N4-------N5
    |  |   |
    |  N3  |
    | /  \\ |
    |/    \\|
   N1------N2
```

# Standard Corner Node Positions
1-3: Bottom triangle
4-6: Top triangle (directly above 1-3)

# Topology Properties
- Dimension: 3
- Corner nodes: 6
- Edges: 9
- Faces: 5 (2 triangles + 3 quads)

**Zero allocation:** All functions return compile-time sized tuples.

See also: [`AbstractTopology`](@ref), [`Pyramid`](@ref)
"""
struct Wedge <: AbstractTopology end

dim(::Wedge) = 3

"""
    reference_coordinates(::Wedge)

Get corner node positions for Wedge (6 vertices).
"""
function reference_coordinates(::Wedge)
    return (
        (0.0, 0.0, -1.0),  # Node 1: Bottom triangle
        (1.0, 0.0, -1.0),  # Node 2: Bottom triangle
        (0.0, 1.0, -1.0),  # Node 3: Bottom triangle
        (0.0, 0.0, 1.0),  # Node 4: Top triangle
        (1.0, 0.0, 1.0),  # Node 5: Top triangle
        (0.0, 1.0, 1.0),  # Node 6: Top triangle
    )
end

"""
    edges(::Wedge)

Edge connectivity for wedge (corner nodes).
"""
function edges(::Wedge)
    return (
        (1, 2), (2, 3), (3, 1),  # Bottom triangle edges
        (4, 5), (5, 6), (6, 4),  # Top triangle edges
        (1, 4), (2, 5), (3, 6),  # Vertical edges
    )
end

"""
    faces(::Wedge)

Face connectivity for wedge (2 triangular + 3 quadrilateral faces).
"""
function faces(::Wedge)
    return (
        (1, 3, 2),        # Face 1: Bottom triangle
        (4, 5, 6),        # Face 2: Top triangle
        (1, 2, 5, 4),     # Face 3: Quad
        (2, 3, 6, 5),     # Face 4: Quad
        (3, 1, 4, 6),     # Face 5: Quad
    )
end

# ============================================================================
# Deprecated aliases (for backwards compatibility)
# ============================================================================

"""
    Wedge6

**DEPRECATED:** Backward compatibility alias. Use `Wedge` with `Lagrange{Wedge, 1}`.

This alias allows old code to work:
```julia
# Old style (still works)
element = Element(Wedge6, (1, 2, 3, 4, 5, 6))
# Internally converted to:
element = Element(Wedge, (1, 2, 3, 4, 5, 6))  # Infers Lagrange{Wedge, 1}
```

New code should use explicit topology + basis:
```julia
element = Element(Lagrange{Wedge, 1}, (1, 2, 3, 4, 5, 6))
```
"""
const Wedge6 = Wedge

"""
    Wedge15

**DEPRECATED:** Backward compatibility alias. Use `Wedge` with `Lagrange{Wedge, 2}`.

This alias allows old code to work:
```julia
# Old style (still works)
element = Element(Wedge15, (1, 2, ..., 15))
# Internally converted to:
element = Element(Wedge, (1, 2, ..., 15))  # Infers Lagrange{Wedge, 2}
```

New code should use explicit topology + basis:
```julia
element = Element(Lagrange{Wedge, 2}, (1, 2, ..., 15))
```
"""
const Wedge15 = Wedge  # Same topology! Node count from basis.
