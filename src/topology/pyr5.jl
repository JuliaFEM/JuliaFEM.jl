# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE

"""
    Pyramid <: AbstractTopology

Pyramid element topology (3D mixed).

Reference element in 3D parametric space with square base at z=-1
and apex at (0,0,1).

# Node numbering
```
       N5
       /\\
      /  \\
     /    \\
    /      \\
   /        \\
  N4--------N3
  |          |
  |          |
  N1--------N2
```

Base nodes in [-1,1]² at z=-1, apex at (0,0,1).

**Note:** This uses Code Aster convention (from lagrange_pyramids.jl).

**Node count:** Determined by basis function degree:
- Lagrange{Pyramid, 1}: 5 nodes
- Lagrange{Pyramid, 2}: 13 nodes
- Lagrange{Pyramid, 3}: 29 nodes

**Zero allocation:** All functions return compile-time sized tuples.
"""
struct Pyramid <: AbstractTopology end

dim(::Pyramid) = 3

# Backwards compatibility alias
const Pyr5 = Pyramid

"""
    reference_coordinates(::Pyramid) -> NTuple{5, NTuple{3, Float64}}

Reference coordinates for pyramid (Code Aster convention):
- Base nodes: (-1,-1,-1), (1,-1,-1), (1,1,-1), (-1,1,-1)
- Apex: (0,0,1)

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
reference_coordinates(::Pyramid) = (
    (-1.0, -1.0, -1.0), # N1
    (1.0, -1.0, -1.0), # N2
    (1.0, 1.0, -1.0), # N3
    (-1.0, 1.0, -1.0), # N4
    (0.0, 0.0, 1.0), # N5 (apex)
)

"""
    edges(::Pyramid) -> NTuple{8, Tuple{Int, Int}}

Edge connectivity for pyramid (4 base edges + 4 edges to apex = 8 edges).

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
edges(::Pyramid) = (
    (1, 2), (2, 3), (3, 4), (4, 1),  # Base
    (1, 5), (2, 5), (3, 5), (4, 5),  # Edges to apex
)

"""
    faces(::Pyramid) -> NTuple{5, Tuple{Vararg{Int}}}

Face connectivity for pyramid:
- 1 quadrilateral base
- 4 triangular faces

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
faces(::Pyramid) = (
    (1, 4, 3, 2), # Base (quad)
    (1, 2, 5),    # Triangle
    (2, 3, 5),    # Triangle
    (3, 4, 5),    # Triangle
    (4, 1, 5),    # Triangle
)
