# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE

"""
    Tetrahedron <: AbstractTopology

Tetrahedral element topology (3D simplex).

**Important:** This type defines ONLY the geometric shape. Node count is determined
by the interpolation scheme (basis functions):
- `Lagrange{Tetrahedron, 1}` → 4 nodes (P1, linear)
- `Lagrange{Tetrahedron, 2}` → 10 nodes (P2, quadratic)
- `Lagrange{Tetrahedron, 3}` → 20 nodes (P3, cubic)

# Reference Element
```
      N4
      /|\\
     / | \\
    /  |  \\
   /   |   \\
  N1---+----N3
   \\  /
    \\/
    N2
```

# Standard Corner Node Positions
1. (0, 0, 0) - Origin
2. (1, 0, 0) - Along ξ-axis
3. (0, 1, 0) - Along η-axis
4. (0, 0, 1) - Along ζ-axis

# Topology Properties
- Dimension: 3
- Corner nodes: 4
- Edges: 6
- Faces: 4 (triangular)

# Typical Usage
```julia
julia> topology = Tetrahedron()
julia> dim(topology)
3
julia> reference_coordinates(topology)  # Corner nodes only
((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

julia> basis = Lagrange{Tetrahedron, 1}()
julia> nnodes(basis)  # Linear: 4 nodes
4

julia> basis = Lagrange{Tetrahedron, 2}()
julia> nnodes(basis)  # Quadratic: 10 nodes (corners + edge midpoints)
10
```

**Zero allocation:** All functions return compile-time sized tuples.

See also: [`AbstractTopology`](@ref), [`Hexahedron`](@ref), [`Lagrange`](@ref)
"""
struct Tetrahedron <: AbstractTopology end

dim(::Tetrahedron) = 3

"""
    reference_coordinates(::Tetrahedron)

Get corner node positions for Tetrahedron (4 vertices in parametric space).
"""
function reference_coordinates(::Tetrahedron)
    return (
        (0.0, 0.0, 0.0),  # Node 1: Origin
        (1.0, 0.0, 0.0),  # Node 2: Along ξ-axis
        (0.0, 1.0, 0.0),  # Node 3: Along η-axis
        (0.0, 0.0, 1.0),  # Node 4: Along ζ-axis
    )
end

"""
    edges(::Tetrahedron)

Edge connectivity for tetrahedron (corner nodes).
"""
function edges(::Tetrahedron)
    return (
        (1, 2),  # Edge 1
        (2, 3),  # Edge 2
        (3, 1),  # Edge 3
        (1, 4),  # Edge 4
        (2, 4),  # Edge 5
        (3, 4),  # Edge 6
    )
end

"""
    faces(::Tetrahedron)

Face connectivity for tetrahedron (triangular faces, corner nodes).
"""
function faces(::Tetrahedron)
    return (
        (1, 3, 2),  # Face 1: Base (looking from above)
        (1, 2, 4),  # Face 2
        (2, 3, 4),  # Face 3
        (3, 1, 4),  # Face 4
    )
end

# ============================================================================
# Deprecated aliases (for backwards compatibility)
# ============================================================================

"""
    Tet4

**DEPRECATED:** Backward compatibility alias. Use `Tetrahedron` with `Lagrange{Tetrahedron, 1}`.

The old `Tet4` conflated topology (tetrahedron) with node count (4).
In the new architecture:
- Topology defines geometric shape only
- Basis functions determine node count

This alias allows old code to work:
```julia
# Old style (still works)
element = Element(Tet4, (1, 2, 3, 4))
# Internally converted to:
element = Element(Tetrahedron, (1, 2, 3, 4))  # Infers Lagrange{Tetrahedron, 1}
```

New code should use explicit topology + basis:
```julia
element = Element(Lagrange{Tetrahedron, 1}, (1, 2, 3, 4))
```
"""
const Tet4 = Tetrahedron

"""
    Tet10

**DEPRECATED:** Backward compatibility alias. Use `Tetrahedron` with `Lagrange{Tetrahedron, 2}`.

This alias allows old code to work:
```julia
# Old style (still works)
element = Element(Tet10, (1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
# Internally converted to:
element = Element(Tetrahedron, (1, 2, 3, 4, 5, 6, 7, 8, 9, 10))  # Infers Lagrange{Tetrahedron, 2}
```

New code should use explicit topology + basis:
```julia
element = Element(Lagrange{Tetrahedron, 2}, (1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
```
"""
const Tet10 = Tetrahedron  # Same topology! Node count from basis.
