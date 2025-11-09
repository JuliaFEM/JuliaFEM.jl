# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE

"""
    Tetrahedron <: AbstractTopology

Tetrahedral element topology (3D simplex).

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

**Note:** Node count determined by basis function degree:
- Lagrange{Tetrahedron, 1}: 4 nodes (P1)
- Lagrange{Tetrahedron, 2}: 10 nodes (P2)

**Zero allocation:** All functions return compile-time sized tuples.
"""
struct Tetrahedron <: AbstractTopology end

dim(::Tetrahedron) = 3

# Backwards compatibility alias
const Tet4 = Tetrahedron

"""
    nnodes(::Tetrahedron) -> Int

Number of corner nodes for a tetrahedron element (4).

**Note:** In the new architecture, actual node count depends on basis degree.
This returns the number of corner nodes for backwards compatibility.
"""
nnodes(::Tetrahedron) = 4

"""
    reference_coordinates(::Tetrahedron) -> NTuple{4, NTuple{3, Float64}}

Reference coordinates for tetrahedron vertices:
(0,0,0), (1,0,0), (0,1,0), (0,0,1).

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
reference_coordinates(::Tetrahedron) = (
    (0.0, 0.0, 0.0), # N1
    (1.0, 0.0, 0.0), # N2
    (0.0, 1.0, 0.0), # N3
    (0.0, 0.0, 1.0), # N4
)

"""
    edges(::Tetrahedron) -> NTuple{6, Tuple{Int, Int}}

Edge connectivity for tetrahedron (6 edges).

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
edges(::Tetrahedron) = (
    (1, 2), (2, 3), (3, 1),  # Base triangle
    (1, 4), (2, 4), (3, 4),  # Edges to apex
)

"""
    faces(::Tetrahedron) -> NTuple{4, NTuple{3, Int}}

Face connectivity for tetrahedron (4 triangular faces).

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
faces(::Tetrahedron) = (
    (1, 3, 2), # Base
    (1, 2, 4), # Front
    (2, 3, 4), # Right
    (3, 1, 4), # Left
)
