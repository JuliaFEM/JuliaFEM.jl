# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE

"""
    Segment <: AbstractTopology

Linear segment/line element topology (1D).

Reference element in 1D parametric space [-1, 1].

# Node numbering
```
 N1 -------- N2
-1           +1
```

**Note:** Node count determined by basis function degree:
- Lagrange{Segment, 1}: 2 nodes (linear)
- Lagrange{Segment, 2}: 3 nodes (quadratic)

**Zero allocation:** All functions return compile-time sized tuples.
"""
struct Segment <: AbstractTopology end

dim(::Segment) = 1

# Backwards compatibility alias
const Seg2 = Segment

"""
    reference_coordinates(::Segment) -> NTuple{2, NTuple{1, Float64}}

Reference coordinates for segment endpoints: (-1.0,) and (1.0,).

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
reference_coordinates(::Segment) = ((-1.0,), (1.0,))

"""
    edges(::Segment) -> NTuple{1, Tuple{Int, Int}}

Edge connectivity for segment (the segment itself).

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
edges(::Segment) = ((1, 2),)

"""
    faces(::Segment) -> NTuple{0, Tuple{}}

Faces for 1D element (none in 1D).

**Zero allocation:** Returns empty tuple (stack allocated).
"""
faces(::Segment) = ()
