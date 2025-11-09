# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE

"""
    Tri6 <: AbstractTopology

6-node quadratic triangle element.

Reference element in 2D parametric space with vertices at (0,0), (1,0), (0,1)
and midpoint nodes on each edge.

# Node numbering
```
N3
 |\\
 | \\
N6  N5
 |   \\
 |    \\
N1--N4--N2
```

**Zero allocation:** All functions return compile-time sized tuples.
"""
struct Tri6 <: AbstractTopology end

nnodes(::Tri6) = 6
dim(::Tri6) = 2

"""
    reference_coordinates(::Tri6) -> NTuple{6, NTuple{2, Float64}}

Reference coordinates for 6-node triangle:
- Corner nodes: (0,0), (1,0), (0,1)
- Edge nodes: (0.5,0), (0.5,0.5), (0,0.5)

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
reference_coordinates(::Tri6) = (
    (0.0, 0.0),  # N1
    (1.0, 0.0),  # N2
    (0.0, 1.0),  # N3
    (0.5, 0.0),  # N4
    (0.5, 0.5),  # N5
    (0.0, 0.5),  # N6
)

"""
    edges(::Tri6) -> NTuple{3, Tuple{Int, Int}}

Edge connectivity for triangle (corner nodes only).

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
edges(::Tri6) = ((1, 2), (2, 3), (3, 1))

"""
    faces(::Tri6) -> NTuple{1, NTuple{6, Int}}

Face connectivity (all nodes) for 2D element.

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
faces(::Tri6) = ((1, 2, 3, 4, 5, 6),)
