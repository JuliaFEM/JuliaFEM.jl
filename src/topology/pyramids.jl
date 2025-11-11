# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE

"""
    Pyramid <: AbstractTopology

Pyramidal element topology (3D).

**Important:** This type defines ONLY the geometric shape. Node count is determined
by the interpolation scheme (basis functions):
- `Lagrange{Pyramid, 1}` → 5 nodes (linear)
- `Lagrange{Pyramid, 2}` → 13 nodes (quadratic, with edge midpoints)

# Reference Element
```
        N5 (apex)
        /|\\
       / | \\
      /  |  \\
     /   |   \\
    N4---+---N3
    |    |   |
    |    N1  |
    |   /    |
    |  /     |
    | /      |
    N1------N2
```

# Standard Corner Node Positions
1-4: Square base in z=0 plane
5: Apex

# Topology Properties
- Dimension: 3
- Corner nodes: 5
- Edges: 8
- Faces: 5 (1 quad + 4 triangles)

**Zero allocation:** All functions return compile-time sized tuples.

See also: [`AbstractTopology`](@ref), [`Wedge`](@ref)
"""
struct Pyramid <: AbstractTopology end

dim(::Pyramid) = 3

"""
    reference_coordinates(::Pyramid)

Get corner node positions for Pyramid (5 vertices).
"""
function reference_coordinates(::Pyramid)
    return (
        (-1.0, -1.0, 0.0),  # Node 1: Base corner
        (1.0, -1.0, 0.0),  # Node 2: Base corner
        (1.0, 1.0, 0.0),  # Node 3: Base corner
        (-1.0, 1.0, 0.0),  # Node 4: Base corner
        (0.0, 0.0, 1.0),  # Node 5: Apex
    )
end

"""
    edges(::Pyramid)

Edge connectivity for pyramid (corner nodes).
"""
function edges(::Pyramid)
    return (
        (1, 2), (2, 3), (3, 4), (4, 1),  # Base edges
        (1, 5), (2, 5), (3, 5), (4, 5),  # Edges to apex
    )
end

"""
    faces(::Pyramid)

Face connectivity for pyramid (1 quad base + 4 triangular faces).
"""
function faces(::Pyramid)
    return (
        (1, 4, 3, 2),  # Face 1: Quadrilateral base
        (1, 2, 5),     # Face 2: Triangle
        (2, 3, 5),     # Face 3: Triangle
        (3, 4, 5),     # Face 4: Triangle
        (4, 1, 5),     # Face 5: Triangle
    )
end

# ============================================================================
# Deprecated aliases (for backwards compatibility)
# ============================================================================

"""
    Pyr5

**DEPRECATED:** Backward compatibility alias. Use `Pyramid` with `Lagrange{Pyramid, 1}`.

This alias allows old code to work:
```julia
# Old style (still works)
element = Element(Pyr5, (1, 2, 3, 4, 5))
# Internally converted to:
element = Element(Pyramid, (1, 2, 3, 4, 5))  # Infers Lagrange{Pyramid, 1}
```

New code should use explicit topology + basis:
```julia
element = Element(Lagrange{Pyramid, 1}, (1, 2, 3, 4, 5))
```
"""
const Pyr5 = Pyramid
