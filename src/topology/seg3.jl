# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE

"""
    Seg3 <: AbstractTopology

3-node quadratic segment/line element.

Reference element in 1D parametric space [-1, 1] with midpoint node.

# Node numbering
```
 N1 ---- N3 ---- N2
-1       0       +1
```

**Zero allocation:** All functions return compile-time sized tuples.
"""
struct Seg3 <: AbstractTopology end

nnodes(::Seg3) = 3
dim(::Seg3) = 1

"""
    reference_coordinates(::Seg3) -> NTuple{3, NTuple{1, Float64}}

Reference coordinates for 3-node segment: (-1.0,), (1.0,), (0.0,).

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
reference_coordinates(::Seg3) = ((-1.0,), (1.0,), (0.0,))

"""
    edges(::Seg3) -> NTuple{1, Tuple{Int, Int}}

Edge connectivity for segment (the segment itself, corner nodes only).

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
edges(::Seg3) = ((1, 2),)

"""
    faces(::Seg3) -> NTuple{0, Tuple{}}

Faces for 1D element (none in 1D).

**Zero allocation:** Returns empty tuple (stack allocated).
"""
faces(::Seg3) = ()
