# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE

"""
    Seg2 <: AbstractTopology

2-node linear segment/line element.

Reference element in 1D parametric space [-1, 1].

# Node numbering
```
 N1 -------- N2
-1           +1
```

**Zero allocation:** All functions return compile-time sized tuples.
"""
struct Seg2 <: AbstractTopology end

nnodes(::Seg2) = 2
dim(::Seg2) = 1

"""
    reference_coordinates(::Seg2) -> NTuple{2, NTuple{1, Float64}}

Reference coordinates for 2-node segment: (-1.0,) and (1.0,).

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
reference_coordinates(::Seg2) = ((-1.0,), (1.0,))

"""
    edges(::Seg2) -> NTuple{1, Tuple{Int, Int}}

Edge connectivity for segment (the segment itself).

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
edges(::Seg2) = ((1, 2),)

"""
    faces(::Seg2) -> NTuple{0, Tuple{}}

Faces for 1D element (none in 1D).

**Zero allocation:** Returns empty tuple (stack allocated).
"""
faces(::Seg2) = ()
