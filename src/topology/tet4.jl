# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE

"""
    Tet4 <: AbstractTopology

4-node linear tetrahedral element.

Reference element in 3D parametric space with vertices at (0,0,0), (1,0,0), (0,1,0), (0,0,1).

# Node numbering
```
      N4
      /|\\
     / | \\
    /  |  \\
   /   |   \\
  /    |    \\
N1-----|-----N3
  \\    |    /
   \\   |   /
    \\  |  /
     \\ | /
      \\|/
       N2
```

**Zero allocation:** All functions return compile-time sized tuples.
"""
struct Tet4 <: AbstractTopology end

nnodes(::Tet4) = 4
dim(::Tet4) = 3

"""
    reference_coordinates(::Tet4) -> NTuple{4, NTuple{3, Float64}}

Reference coordinates for 4-node tetrahedron:
(0,0,0), (1,0,0), (0,1,0), (0,0,1).

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
reference_coordinates(::Tet4) = (
    (0.0, 0.0, 0.0), # N1
    (1.0, 0.0, 0.0), # N2
    (0.0, 1.0, 0.0), # N3
    (0.0, 0.0, 1.0), # N4
)

"""
    edges(::Tet4) -> NTuple{6, Tuple{Int, Int}}

Edge connectivity for tetrahedron (6 edges).

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
edges(::Tet4) = (
    (1, 2), (2, 3), (3, 1),  # Base triangle
    (1, 4), (2, 4), (3, 4),  # Edges to apex
)

"""
    faces(::Tet4) -> NTuple{4, NTuple{3, Int}}

Face connectivity for tetrahedron (4 triangular faces).

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
faces(::Tet4) = (
    (1, 3, 2), # Base
    (1, 2, 4), # Front
    (2, 3, 4), # Right
    (3, 1, 4), # Left
)
