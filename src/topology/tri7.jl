# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE

"""
    Tri7 <: AbstractTopology

7-node quadratic triangle element with center node.

Reference element in 2D parametric space with vertices at (0,0), (1,0), (0,1),
midpoint nodes on each edge, and center node.

# Node numbering
```
N3
 |\\
 | \\
N6 N7 N5
 |   \\
 |    \\
N1--N4--N2
```

**Zero allocation:** All functions return compile-time sized tuples.
"""
struct Tri7 <: AbstractTopology end

nnodes(::Tri7) = 7
dim(::Tri7) = 2

"""
    reference_coordinates(::Tri7) -> NTuple{7, NTuple{2, Float64}}

Reference coordinates for 7-node triangle:
- Corner nodes: (0,0), (1,0), (0,1)
- Edge nodes: (0.5,0), (0.5,0.5), (0,0.5)
- Center node: (1/3, 1/3)

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
reference_coordinates(::Tri7) = (
    (0.0, 0.0),    # N1
    (1.0, 0.0),    # N2
    (0.0, 1.0),    # N3
    (0.5, 0.0),    # N4
    (0.5, 0.5),    # N5
    (0.0, 0.5),    # N6
    (1 / 3, 1 / 3),    # N7 (center)
)

"""
    edges(::Tri7) -> NTuple{3, Tuple{Int, Int}}

Edge connectivity for triangle (corner nodes only).

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
edges(::Tri7) = ((1, 2), (2, 3), (3, 1))

"""
    faces(::Tri7) -> NTuple{1, NTuple{7, Int}}

Face connectivity (all nodes) for 2D element.

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
faces(::Tri7) = ((1, 2, 3, 4, 5, 6, 7),)
