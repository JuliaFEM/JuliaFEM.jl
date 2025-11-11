# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE

"""
    Segment <: AbstractTopology

Linear segment/line element topology (1D).

Reference element in 1D parametric space [-1, 1].

# Node numbering (corners only)
```
 N1 -------- N2
-1           +1
```

**Important:** This type defines ONLY the geometric shape (1D line segment).
Node count is determined by the basis functions:
- `Lagrange{Segment, 1}` → 2 nodes (linear)
- `Lagrange{Segment, 2}` → 3 nodes (quadratic, adds midpoint)
- `Lagrange{Segment, 3}` → 4 nodes (cubic)

# Topology Properties
- Dimension: 1
- Corner nodes: 2
- Edges: 1 (the element itself)
- Faces: 0 (none in 1D)

# Typical Usage
```julia
julia> topology = Segment()
julia> dim(topology)
1
julia> reference_coordinates(topology)  # Corner nodes only
((-1.0,), (1.0,))

julia> basis = Lagrange{Segment, 1}()
julia> nnodes(basis)  # Node count from BASIS
2

julia> basis = Lagrange{Segment, 2}()
julia> nnodes(basis)  # Quadratic has 3 nodes
3
```

**Zero allocation:** All functions return compile-time sized tuples.

See also: [`AbstractTopology`](@ref), [`Lagrange`](@ref)
"""
struct Segment <: AbstractTopology end

dim(::Segment) = 1

"""
    reference_coordinates(::Segment) -> NTuple{2, NTuple{1, Float64}}

Reference coordinates for segment corner nodes: (-1.0,) and (1.0,).

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
reference_coordinates(::Segment) = ((-1.0,), (1.0,))

"""
    edges(::Segment) -> NTuple{1, Tuple{Int, Int}}

Edge connectivity for segment (the segment itself, corner nodes).

**Zero allocation:** Returns tuple of tuples (stack allocated).
"""
edges(::Segment) = ((1, 2),)

"""
    faces(::Segment) -> NTuple{0, Tuple{}}

Faces for 1D element (none in 1D).

**Zero allocation:** Returns empty tuple (stack allocated).
"""
faces(::Segment) = ()

# ============================================================================
# Deprecated aliases (for backwards compatibility)
# ============================================================================

"""
    Seg2

**DEPRECATED:** Backward compatibility alias. Use `Segment` with `Lagrange{Segment, 1}`.

The old `Seg2` conflated topology (segment) with node count (2).
In the new architecture:
- Topology defines geometric shape only
- Basis functions determine node count

This alias allows old code to work:
```julia
# Old style (still works)
element = Element(Seg2, (1, 2))
# Internally converted to:
element = Element(Segment, (1, 2))  # Infers Lagrange{Segment, 1} from 2 nodes
```

New code should use explicit topology + basis:
```julia
element = Element(Lagrange{Segment, 1}, (1, 2))
```
"""
const Seg2 = Segment

"""
    Seg3

**DEPRECATED:** Backward compatibility alias. Use `Segment` with `Lagrange{Segment, 2}`.

The old `Seg3` conflated topology (segment) with node count (3).

This alias allows old code to work:
```julia
# Old style (still works)
element = Element(Seg3, (1, 2, 3))
# Internally converted to:
element = Element(Segment, (1, 2, 3))  # Infers Lagrange{Segment, 2} from 3 nodes
```

New code should use explicit topology + basis:
```julia
element = Element(Lagrange{Segment, 2}, (1, 2, 3))
```
"""
const Seg3 = Segment  # Yes, same topology! Node count from basis.
