# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE

"""
    Hexahedron <: AbstractTopology

Hexahedral element topology (3D tensor product).

**Important:** This type defines ONLY the geometric shape. Node count is determined
by the interpolation scheme (basis functions):
- `Lagrange{Hexahedron, 1}` → 8 nodes (Q1, trilinear)
- `Serendipity{Hexahedron, 2}` → 20 nodes (Q2, no interior nodes)
- `Lagrange{Hexahedron, 2}` → 27 nodes (Q2, full tensor product)

# Reference Element
```
      N8-------N7
      /|      /|
     / |     / |
   N5-------N6 |
    | N4----|--N3
    | /     | /
    |/      |/
   N1-------N2
```

# Standard Corner Node Positions (in [-1,1]³)
1. (-1, -1, -1)
2. ( 1, -1, -1)
3. ( 1,  1, -1)
4. (-1,  1, -1)
5. (-1, -1,  1)
6. ( 1, -1,  1)
7. ( 1,  1,  1)
8. (-1,  1,  1)

# Topology Properties
- Dimension: 3
- Corner nodes: 8
- Edges: 12
- Faces: 6 (quadrilateral)

# Typical Usage
```julia
julia> topology = Hexahedron()
julia> dim(topology)
3
julia> reference_coordinates(topology)  # Corner nodes only
((-1.0, -1.0, -1.0), (1.0, -1.0, -1.0), ...)

julia> basis = Lagrange{Hexahedron, 1}()
julia> nnodes(basis)  # Trilinear: 8 nodes
8

julia> basis = Serendipity{Hexahedron, 2}()
julia> nnodes(basis)  # Serendipity: 20 nodes (no interior)
20

julia> basis = Lagrange{Hexahedron, 2}()
julia> nnodes(basis)  # Full Lagrange: 27 nodes (with interior)
27
```

**Zero allocation:** All functions return compile-time sized tuples.

See also: [`AbstractTopology`](@ref), [`Tetrahedron`](@ref), [`Lagrange`](@ref), [`Serendipity`](@ref)
"""
struct Hexahedron <: AbstractTopology end

dim(::Hexahedron) = 3

"""
    reference_coordinates(::Hexahedron)

Get corner node positions for Hexahedron (8 vertices in [-1,1]³).
"""
function reference_coordinates(::Hexahedron)
    return (
        (-1.0, -1.0, -1.0),  # Node 1
        (1.0, -1.0, -1.0),  # Node 2
        (1.0, 1.0, -1.0),  # Node 3
        (-1.0, 1.0, -1.0),  # Node 4
        (-1.0, -1.0, 1.0),  # Node 5
        (1.0, -1.0, 1.0),  # Node 6
        (1.0, 1.0, 1.0),  # Node 7
        (-1.0, 1.0, 1.0),  # Node 8
    )
end

"""
    edges(::Hexahedron)

Edge connectivity for hexahedron (corner nodes).
"""
function edges(::Hexahedron)
    return (
        (1, 2), (2, 3), (3, 4), (4, 1),  # Bottom face edges
        (5, 6), (6, 7), (7, 8), (8, 5),  # Top face edges
        (1, 5), (2, 6), (3, 7), (4, 8),  # Vertical edges
    )
end

"""
    faces(::Hexahedron)

Face connectivity for hexahedron (quadrilateral faces, corner nodes).
"""
function faces(::Hexahedron)
    return (
        (1, 4, 3, 2),  # Face 1: Bottom (-z)
        (1, 2, 6, 5),  # Face 2: Front (-y)
        (2, 3, 7, 6),  # Face 3: Right (+x)
        (3, 4, 8, 7),  # Face 4: Back (+y)
        (4, 1, 5, 8),  # Face 5: Left (-x)
        (5, 6, 7, 8),  # Face 6: Top (+z)
    )
end

# ============================================================================
# Deprecated aliases (for backwards compatibility)
# ============================================================================

"""
    Hex8

**DEPRECATED:** Backward compatibility alias. Use `Hexahedron` with `Lagrange{Hexahedron, 1}`.

The old `Hex8` conflated topology (hexahedron) with node count (8).
In the new architecture:
- Topology defines geometric shape only
- Basis functions determine node count

This alias allows old code to work:
```julia
# Old style (still works)
element = Element(Hex8, (1, 2, 3, 4, 5, 6, 7, 8))
# Internally converted to:
element = Element(Hexahedron, (1, 2, 3, 4, 5, 6, 7, 8))  # Infers Lagrange{Hexahedron, 1}
```

New code should use explicit topology + basis:
```julia
element = Element(Lagrange{Hexahedron, 1}, (1, 2, 3, 4, 5, 6, 7, 8))
```
"""
const Hex8 = Hexahedron

"""
    Hex20

**DEPRECATED:** Backward compatibility alias. Use `Hexahedron` with `Serendipity{Hexahedron, 2}`.

20-node hexahedron with serendipity basis (NO interior nodes).

This alias allows old code to work:
```julia
# Old style (still works)
element = Element(Hex20, (1, 2, ..., 20))
# Internally converted to:
element = Element(Hexahedron, (1, 2, ..., 20))  # Infers Serendipity
```

New code should be explicit about basis family:
```julia
element = Element(Serendipity{Hexahedron, 2}, (1, 2, ..., 20))
```
"""
const Hex20 = Hexahedron  # Same topology! Basis determines node pattern.

"""
    Hex27

**DEPRECATED:** Backward compatibility alias. Use `Hexahedron` with `Lagrange{Hexahedron, 2}`.

27-node hexahedron with full tensor product basis (WITH interior nodes).

This alias allows old code to work:
```julia
# Old style (still works)
element = Element(Hex27, (1, 2, ..., 27))
# Internally converted to:
element = Element(Hexahedron, (1, 2, ..., 27))  # Infers Lagrange
```

New code should be explicit:
```julia
element = Element(Lagrange{Hexahedron, 2}, (1, 2, ..., 27))
```
"""
const Hex27 = Hexahedron  # Same topology! Basis determines node pattern.
