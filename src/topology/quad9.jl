# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE

"""
    Quad9 <: AbstractTopology

9-node quadratic quadrilateral element.

Reference element in 2D parametric space [-1, 1]² with corner, edge, and center nodes.

# Node numbering
```
N4----N7----N3
|           |
N8    N9    N6
|           |
N1----N5----N2
```

**Zero allocation:** All functions return compile-time sized tuples.
"""
struct Quad9 <: AbstractTopology end

nnodes(::Quad9) = 9
dim(::Quad9) = 2

"""
    reference_coordinates(::Quad9) -> NTuple{9, NTuple{2, Float64}}

Reference coordinates for 9-node quadrilateral:
- Corner nodes: (-1,-1), (1,-1), (1,1), (-1,1)
- Edge nodes: (0,-1), (1,0), (0,1), (-1,0)
- Center node: (0,0)

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
reference_coordinates(::Quad9) = (
    (-1.0, -1.0), # N1
    (1.0, -1.0), # N2
    (1.0, 1.0), # N3
    (-1.0, 1.0), # N4
    (0.0, -1.0), # N5
    (1.0, 0.0), # N6
    (0.0, 1.0), # N7
    (-1.0, 0.0), # N8
    (0.0, 0.0), # N9 (center)
)

"""
    edges(::Quad9) -> NTuple{4, Tuple{Int, Int}}

Edge connectivity for quadrilateral (corner nodes only).

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
edges(::Quad9) = ((1, 2), (2, 3), (3, 4), (4, 1))

"""
    faces(::Quad9) -> NTuple{1, NTuple{9, Int}}

Face connectivity (all nodes) for 2D element.

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
faces(::Quad9) = ((1, 2, 3, 4, 5, 6, 7, 8, 9),)
