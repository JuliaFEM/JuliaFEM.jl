# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE

"""
    Wedge15 <: AbstractTopology

15-node quadratic prismatic/wedge element.

Reference element in 3D with triangular cross-section in (u,v) plane
extruded along w direction, with edge and mid-plane nodes.

# Node numbering
```
      N6
      /\\
    N12 N11
    /    \\
   N4-N10-N5
   |      |
  N15     N14
   |      |
  N3      |
   |\\     |
   N9 N8  |
   |  \\   N13
   |   \\  |
   |    \\ |
   |     \\|
  N1--N7--N2
```

**Zero allocation:** All functions return compile-time sized tuples.
"""
struct Wedge15 <: AbstractTopology end

nnodes(::Wedge15) = 15
dim(::Wedge15) = 3

"""
    reference_coordinates(::Wedge15) -> NTuple{15, NTuple{3, Float64}}

Reference coordinates for 15-node wedge:
- Corner nodes (6): bottom and top triangles
- Edge nodes (9): 3 on bottom, 3 on top, 3 on vertical edges

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
reference_coordinates(::Wedge15) = (
    (0.0, 0.0, -1.0), # N1
    (1.0, 0.0, -1.0), # N2
    (0.0, 1.0, -1.0), # N3
    (0.0, 0.0,  1.0), # N4
    (1.0, 0.0,  1.0), # N5
    (0.0, 1.0,  1.0), # N6
    (0.5, 0.0, -1.0), # N7  (edge 1-2, bottom)
    (0.5, 0.5, -1.0), # N8  (edge 2-3, bottom)
    (0.0, 0.5, -1.0), # N9  (edge 3-1, bottom)
    (0.5, 0.0,  1.0), # N10 (edge 4-5, top)
    (0.5, 0.5,  1.0), # N11 (edge 5-6, top)
    (0.0, 0.5,  1.0), # N12 (edge 6-4, top)
    (0.0, 0.0,  0.0), # N13 (edge 1-4, vertical)
    (1.0, 0.0,  0.0), # N14 (edge 2-5, vertical)
    (0.0, 1.0,  0.0), # N15 (edge 3-6, vertical)
)

"""
    edges(::Wedge15) -> NTuple{9, Tuple{Int, Int}}

Edge connectivity for wedge (corner nodes only).

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
edges(::Wedge15) = (
    (1, 2), (2, 3), (3, 1),  # Bottom triangle
    (4, 5), (5, 6), (6, 4),  # Top triangle
    (1, 4), (2, 5), (3, 6),  # Vertical edges
)

"""
    faces(::Wedge15) -> NTuple{5, Tuple{Vararg{Int}}}

Face connectivity for wedge (corner nodes only):
- 2 triangular faces
- 3 quadrilateral faces

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
faces(::Wedge15) = (
    (1, 3, 2),    # Bottom triangle
    (4, 5, 6),    # Top triangle
    (1, 2, 5, 4), # Side quad
    (2, 3, 6, 5), # Side quad
    (3, 1, 4, 6), # Side quad
)
