# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE

"""
    Pyr5 <: AbstractTopology

5-node linear pyramid element.

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

**Zero allocation:** All functions return compile-time sized tuples.
"""
struct Pyr5 <: AbstractTopology end

nnodes(::Pyr5) = 5
dim(::Pyr5) = 3

"""
    reference_coordinates(::Pyr5) -> NTuple{5, NTuple{3, Float64}}

Reference coordinates for 5-node pyramid (Code Aster convention):
- Base nodes: (-1,-1,-1), (1,-1,-1), (1,1,-1), (-1,1,-1)
- Apex: (0,0,1)

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
reference_coordinates(::Pyr5) = (
    (-1.0, -1.0, -1.0), # N1
    (1.0, -1.0, -1.0), # N2
    (1.0, 1.0, -1.0), # N3
    (-1.0, 1.0, -1.0), # N4
    (0.0, 0.0, 1.0), # N5 (apex)
)

"""
    edges(::Pyr5) -> NTuple{8, Tuple{Int, Int}}

Edge connectivity for pyramid (4 base edges + 4 edges to apex = 8 edges).

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
edges(::Pyr5) = (
    (1, 2), (2, 3), (3, 4), (4, 1),  # Base
    (1, 5), (2, 5), (3, 5), (4, 5),  # Edges to apex
)

"""
    faces(::Pyr5) -> NTuple{5, Tuple{Vararg{Int}}}

Face connectivity for pyramid:
- 1 quadrilateral base
- 4 triangular faces

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
faces(::Pyr5) = (
    (1, 4, 3, 2), # Base (quad)
    (1, 2, 5),    # Triangle
    (2, 3, 5),    # Triangle
    (3, 4, 5),    # Triangle
    (4, 1, 5),    # Triangle
)
