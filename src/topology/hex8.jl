# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE

"""
    Hexahedron <: AbstractTopology

Hexahedral element topology (3D tensor product).

Reference element in 3D parametric space [-1, 1]³.

# Node numbering
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

**Note:** Node count determined by basis function degree:
- Lagrange{Hexahedron, 1}: 8 nodes (Q1)
- Lagrange{Hexahedron, 2}: 27 nodes (Q2)

**Zero allocation:** All functions return compile-time sized tuples.
"""
struct Hexahedron <: AbstractTopology end

dim(::Hexahedron) = 3

# Backwards compatibility alias
const Hex8 = Hexahedron

"""
    nnodes(::Hexahedron) -> Int

Number of corner nodes for a hexahedron element (8).

**Note:** In the new architecture, actual node count depends on basis degree.
This returns the number of corner nodes for backwards compatibility.
"""
nnodes(::Hexahedron) = 8

"""
    reference_coordinates(::Hexahedron) -> NTuple{8, NTuple{3, Float64}}

Reference coordinates for hexahedron vertices in [-1,1]³.

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
reference_coordinates(::Hexahedron) = (
    (-1.0, -1.0, -1.0), # N1
    (1.0, -1.0, -1.0), # N2
    (1.0, 1.0, -1.0), # N3
    (-1.0, 1.0, -1.0), # N4
    (-1.0, -1.0, 1.0), # N5
    (1.0, -1.0, 1.0), # N6
    (1.0, 1.0, 1.0), # N7
    (-1.0, 1.0, 1.0), # N8
)

"""
    edges(::Hexahedron) -> NTuple{12, Tuple{Int, Int}}

Edge connectivity for hexahedron (12 edges).

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
edges(::Hexahedron) = (
    (1, 2), (2, 3), (3, 4), (4, 1),  # Bottom face
    (5, 6), (6, 7), (7, 8), (8, 5),  # Top face
    (1, 5), (2, 6), (3, 7), (4, 8),  # Vertical edges
)

"""
    faces(::Hexahedron) -> NTuple{6, NTuple{4, Int}}

Face connectivity for hexahedron (6 quadrilateral faces).

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
faces(::Hexahedron) = (
    (1, 4, 3, 2), # Bottom (-z)
    (5, 6, 7, 8), # Top (+z)
    (1, 2, 6, 5), # Front (-y)
    (3, 4, 8, 7), # Back (+y)
    (2, 3, 7, 6), # Right (+x)
    (1, 5, 8, 4), # Left (-x)
)
