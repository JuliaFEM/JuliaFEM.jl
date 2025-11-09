# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE

"""
    Hex20 <: AbstractTopology

20-node biquadratic hexahedral element.

Reference element in 3D parametric space [-1, 1]³ with corner and edge nodes
(Serendipity family - no face or volume interior nodes).

# Node numbering
```
      N8---N20---N7
      /|        /|
    N16|      N15|
    /  N12   /  N11
   N5---N17-N6   |
    | N4-|--N19--N3
   N13 /   N14  /
    | N9    | N10
    |/      |/
   N1---N18-N2
```

**Zero allocation:** All functions return compile-time sized tuples.
"""
struct Hex20 <: AbstractTopology end

nnodes(::Hex20) = 20
dim(::Hex20) = 3

"""
    reference_coordinates(::Hex20) -> NTuple{20, NTuple{3, Float64}}

Reference coordinates for 20-node hexahedron (Serendipity):
- Corner nodes (8): vertices of [-1,1]³
- Edge nodes (12): midpoints of 12 edges

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
reference_coordinates(::Hex20) = (
    (-1.0, -1.0, -1.0), # N1
    ( 1.0, -1.0, -1.0), # N2
    ( 1.0,  1.0, -1.0), # N3
    (-1.0,  1.0, -1.0), # N4
    (-1.0, -1.0,  1.0), # N5
    ( 1.0, -1.0,  1.0), # N6
    ( 1.0,  1.0,  1.0), # N7
    (-1.0,  1.0,  1.0), # N8
    ( 0.0, -1.0, -1.0), # N9  (edge 1-2)
    ( 1.0,  0.0, -1.0), # N10 (edge 2-3)
    ( 0.0,  1.0, -1.0), # N11 (edge 3-4)
    (-1.0,  0.0, -1.0), # N12 (edge 4-1)
    (-1.0, -1.0,  0.0), # N13 (edge 1-5)
    ( 1.0, -1.0,  0.0), # N14 (edge 2-6)
    ( 1.0,  1.0,  0.0), # N15 (edge 3-7)
    (-1.0,  1.0,  0.0), # N16 (edge 4-8)
    ( 0.0, -1.0,  1.0), # N17 (edge 5-6)
    ( 1.0,  0.0,  1.0), # N18 (edge 6-7)
    ( 0.0,  1.0,  1.0), # N19 (edge 7-8)
    (-1.0,  0.0,  1.0), # N20 (edge 8-5)
)

"""
    edges(::Hex20) -> NTuple{12, Tuple{Int, Int}}

Edge connectivity for hexahedron (corner nodes only).

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
edges(::Hex20) = (
    (1, 2), (2, 3), (3, 4), (4, 1),  # Bottom face
    (5, 6), (6, 7), (7, 8), (8, 5),  # Top face
    (1, 5), (2, 6), (3, 7), (4, 8),  # Vertical edges
)

"""
    faces(::Hex20) -> NTuple{6, NTuple{4, Int}}

Face connectivity for hexahedron (corner nodes of 6 quadrilateral faces).

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
faces(::Hex20) = (
    (1, 4, 3, 2), # Bottom (-z)
    (5, 6, 7, 8), # Top (+z)
    (1, 2, 6, 5), # Front (-y)
    (3, 4, 8, 7), # Back (+y)
    (2, 3, 7, 6), # Right (+x)
    (1, 5, 8, 4), # Left (-x)
)
