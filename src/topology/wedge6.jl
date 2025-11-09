# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE

"""
    Wedge <: AbstractTopology

Wedge/prism element topology (3D triangular extrusion).

Reference element in 3D with triangular cross-section in (u,v) plane
extruded along w direction from -1 to +1.

# Node numbering
```
      N6
      /\\
     /  \\
    /    \\
   N4----N5
   |      |
   |      |
  N3      |
   |\\     |
   | \\    |
   |  \\   |
   |   \\  |
   |    \\ |
   |     \\|
  N1-----N2
```

Triangle base at w=-1, triangle top at w=+1.

**Note:** Node count determined by basis function degree:
- Lagrange{Wedge, 1}: 6 nodes (P1)
- Lagrange{Wedge, 2}: 15 nodes (P2)

**Zero allocation:** All functions return compile-time sized tuples.
"""
struct Wedge <: AbstractTopology end

dim(::Wedge) = 3

# Backwards compatibility alias
const Wedge6 = Wedge

"""
    reference_coordinates(::Wedge) -> NTuple{6, NTuple{3, Float64}}

Reference coordinates for wedge vertices:
- Bottom triangle (w=-1): (0,0,-1), (1,0,-1), (0,1,-1)
- Top triangle (w=+1): (0,0,+1), (1,0,+1), (0,1,+1)

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
reference_coordinates(::Wedge) = (
    (0.0, 0.0, -1.0), # N1
    (1.0, 0.0, -1.0), # N2
    (0.0, 1.0, -1.0), # N3
    (0.0, 0.0, 1.0), # N4
    (1.0, 0.0, 1.0), # N5
    (0.0, 1.0, 1.0), # N6
)

"""
    edges(::Wedge) -> NTuple{9, Tuple{Int, Int}}

Edge connectivity for wedge (3 bottom + 3 top + 3 vertical = 9 edges).

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
edges(::Wedge) = (
    (1, 2), (2, 3), (3, 1),  # Bottom triangle
    (4, 5), (5, 6), (6, 4),  # Top triangle
    (1, 4), (2, 5), (3, 6),  # Vertical edges
)

"""
    faces(::Wedge) -> NTuple{5, Tuple{Vararg{Int}}}

Face connectivity for wedge:
- 2 triangular faces (top and bottom)
- 3 quadrilateral faces (sides)

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
faces(::Wedge) = (
    (1, 3, 2),    # Bottom triangle
    (4, 5, 6),    # Top triangle
    (1, 2, 5, 4), # Side quad
    (2, 3, 6, 5), # Side quad
    (3, 1, 4, 6), # Side quad
)
