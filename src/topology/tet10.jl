# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE

"""
    Tet10 <: AbstractTopology

10-node quadratic tetrahedral element.

Reference element in 3D parametric space with corner nodes at vertices
and midpoint nodes on each edge.

# Node numbering
```
      N4
      /|\\
    N8 | N10
    /  N9  \\
   /   |   \\
  /    |    \\
N1-N5--N7----N3
  \\    |    /
   \\   |   /
    \\ N6  /
     \\ | /
      \\|/
       N2
```

**Zero allocation:** All functions return compile-time sized tuples.
"""
struct Tet10 <: AbstractTopology end

nnodes(::Tet10) = 10
dim(::Tet10) = 3

"""
    reference_coordinates(::Tet10) -> NTuple{10, NTuple{3, Float64}}

Reference coordinates for 10-node tetrahedron:
- Corner nodes: (0,0,0), (1,0,0), (0,1,0), (0,0,1)
- Edge nodes: midpoints of all 6 edges

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
reference_coordinates(::Tet10) = (
    (0.0, 0.0, 0.0), # N1
    (1.0, 0.0, 0.0), # N2
    (0.0, 1.0, 0.0), # N3
    (0.0, 0.0, 1.0), # N4
    (0.5, 0.0, 0.0), # N5  (edge 1-2)
    (0.5, 0.5, 0.0), # N6  (edge 2-3)
    (0.0, 0.5, 0.0), # N7  (edge 3-1)
    (0.0, 0.0, 0.5), # N8  (edge 1-4)
    (0.5, 0.0, 0.5), # N9  (edge 2-4)
    (0.0, 0.5, 0.5), # N10 (edge 3-4)
)

"""
    edges(::Tet10) -> NTuple{6, Tuple{Int, Int}}

Edge connectivity for tetrahedron (corner nodes only).

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
edges(::Tet10) = (
    (1, 2), (2, 3), (3, 1),  # Base triangle
    (1, 4), (2, 4), (3, 4),  # Edges to apex
)

"""
    faces(::Tet10) -> NTuple{4, NTuple{3, Int}}

Face connectivity for tetrahedron (corner nodes of 4 triangular faces).

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
faces(::Tet10) = (
    (1, 3, 2), # Base
    (1, 2, 4), # Front
    (2, 3, 4), # Right
    (3, 1, 4), # Left
)
