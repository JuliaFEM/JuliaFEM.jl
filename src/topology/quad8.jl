# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE

"""
    Quad8 <: AbstractTopology

8-node quadratic quadrilateral element (Serendipity).

Reference element in 2D parametric space [-1, 1]² with corner and edge nodes
but no center node (Serendipity family).

# Node numbering
```
N4----N7----N3
|           |
N8          N6
|           |
N1----N5----N2
```

**Zero allocation:** All functions return compile-time sized tuples.
"""
struct Quad8 <: AbstractTopology end

nnodes(::Quad8) = 8
dim(::Quad8) = 2

"""
    reference_coordinates(::Quad8) -> NTuple{8, NTuple{2, Float64}}

Reference coordinates for 8-node quadrilateral (Serendipity):
- Corner nodes: (-1,-1), (1,-1), (1,1), (-1,1)
- Edge nodes: (0,-1), (1,0), (0,1), (-1,0)

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
reference_coordinates(::Quad8) = (
    (-1.0, -1.0), # N1
    ( 1.0, -1.0), # N2
    ( 1.0,  1.0), # N3
    (-1.0,  1.0), # N4
    ( 0.0, -1.0), # N5
    ( 1.0,  0.0), # N6
    ( 0.0,  1.0), # N7
    (-1.0,  0.0), # N8
)

"""
    edges(::Quad8) -> NTuple{4, Tuple{Int, Int}}

Edge connectivity for quadrilateral (corner nodes only).

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
edges(::Quad8) = ((1, 2), (2, 3), (3, 4), (4, 1))

"""
    faces(::Quad8) -> NTuple{1, NTuple{8, Int}}

Face connectivity (all nodes) for 2D element.

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
faces(::Quad8) = ((1, 2, 3, 4, 5, 6, 7, 8),)
