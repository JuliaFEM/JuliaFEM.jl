# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE

"""
    Hex8 <: AbstractTopology

8-node linear hexahedral element.

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

**Zero allocation:** All functions return compile-time sized tuples.
"""
struct Hex8 <: AbstractTopology end

nnodes(::Hex8) = 8
dim(::Hex8) = 3

"""
    reference_coordinates(::Hex8) -> NTuple{8, NTuple{3, Float64}}

Reference coordinates for 8-node hexahedron in [-1,1]³.

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
reference_coordinates(::Hex8) = (
    (-1.0, -1.0, -1.0), # N1
    ( 1.0, -1.0, -1.0), # N2
    ( 1.0,  1.0, -1.0), # N3
    (-1.0,  1.0, -1.0), # N4
    (-1.0, -1.0,  1.0), # N5
    ( 1.0, -1.0,  1.0), # N6
    ( 1.0,  1.0,  1.0), # N7
    (-1.0,  1.0,  1.0), # N8
)

"""
    edges(::Hex8) -> NTuple{12, Tuple{Int, Int}}

Edge connectivity for hexahedron (12 edges).

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
edges(::Hex8) = (
    (1, 2), (2, 3), (3, 4), (4, 1),  # Bottom face
    (5, 6), (6, 7), (7, 8), (8, 5),  # Top face
    (1, 5), (2, 6), (3, 7), (4, 8),  # Vertical edges
)

"""
    faces(::Hex8) -> NTuple{6, NTuple{4, Int}}

Face connectivity for hexahedron (6 quadrilateral faces).

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
faces(::Hex8) = (
    (1, 4, 3, 2), # Bottom (-z)
    (5, 6, 7, 8), # Top (+z)
    (1, 2, 6, 5), # Front (-y)
    (3, 4, 8, 7), # Back (+y)
    (2, 3, 7, 6), # Right (+x)
    (1, 5, 8, 4), # Left (-x)
)
