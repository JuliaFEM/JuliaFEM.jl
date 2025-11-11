# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
    Triangle <: AbstractTopology

Triangular element topology in 2D (reference element geometry).

**Important:** This type defines ONLY the geometric shape. Node count is determined
by the interpolation scheme (basis functions):
- `Lagrange{Triangle, 1}` → 3 nodes (P1, linear)
- `Lagrange{Triangle, 2}` → 6 nodes (P2, quadratic)
- `Lagrange{Triangle, 3}` → 10 nodes (P3, cubic)

# Reference Element
```
     η
     ^
     |
  (0,1) N3
     |  \\
     |    \\
     |      \\
     +---------> ξ
  (0,0)    (1,0)
   N1        N2
```

# Standard Corner Node Positions
1. (0, 0) - Origin
2. (1, 0) - Along ξ-axis
3. (0, 1) - Along η-axis

# Topology Properties
- Dimension: 2
- Corner nodes: 3
- Edges: 3
- Faces: 1 (the element itself in 2D)

# Typical Usage
```julia
julia> topology = Triangle()
julia> dim(topology)
2
julia> reference_coordinates(topology)  # Corner nodes only
((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))

julia> basis = Lagrange{Triangle, 1}()
julia> nnodes(basis)  # Node count from BASIS
3

julia> basis = Lagrange{Triangle, 2}()
julia> nnodes(basis)  # Quadratic has 6 nodes (corners + edge midpoints)
6
```

**Zero allocation:** All functions return compile-time sized tuples.

See also: [`AbstractTopology`](@ref), [`Quadrilateral`](@ref), [`Lagrange`](@ref)
"""
struct Triangle <: AbstractTopology end

dim(::Triangle) = 2

"""
    reference_coordinates(::Triangle)

Get corner node positions for Triangle (3 vertices in parametric space).
"""
function reference_coordinates(::Triangle)
    return (
        (0.0, 0.0),  # Node 1: Origin
        (1.0, 0.0),  # Node 2: Along ξ-axis
        (0.0, 1.0),  # Node 3: Along η-axis
    )
end

"""
    edges(::Triangle)

Edge connectivity for triangle (corner nodes).
"""
function edges(::Triangle)
    return (
        (1, 2),  # Edge 1: Bottom
        (2, 3),  # Edge 2: Right (hypotenuse)
        (3, 1),  # Edge 3: Left
    )
end

"""
    faces(::Triangle)

For 2D elements, the face is the element itself.
"""
faces(::Triangle) = ((1, 2, 3),)

# ============================================================================
# Deprecated aliases (for backwards compatibility)
# ============================================================================

"""
    Tri3

**DEPRECATED:** Backward compatibility alias. Use `Triangle` with `Lagrange{Triangle, 1}`.

The old `Tri3` conflated topology (triangle) with node count (3).
In the new architecture:
- Topology defines geometric shape only
- Basis functions determine node count

This alias allows old code to work:
```julia
# Old style (still works)
element = Element(Tri3, (1, 2, 3))
# Internally converted to:
element = Element(Triangle, (1, 2, 3))  # Infers Lagrange{Triangle, 1} from 3 nodes
```

New code should use explicit topology + basis:
```julia
element = Element(Lagrange{Triangle, 1}, (1, 2, 3))
```
"""
const Tri3 = Triangle

"""
    Tri6

**DEPRECATED:** Backward compatibility alias. Use `Triangle` with `Lagrange{Triangle, 2}`.

This alias allows old code to work:
```julia
# Old style (still works)
element = Element(Tri6, (1, 2, 3, 4, 5, 6))
# Internally converted to:
element = Element(Triangle, (1, 2, 3, 4, 5, 6))  # Infers Lagrange{Triangle, 2} from 6 nodes
```

New code should use explicit topology + basis:
```julia
element = Element(Lagrange{Triangle, 2}, (1, 2, 3, 4, 5, 6))
```
"""
const Tri6 = Triangle  # Same topology! Node count from basis.

"""
    Tri7

**DEPRECATED:** Backward compatibility alias. Use `Triangle` with `Lagrange{Triangle, 2}` + bubble.

7-node triangle with interior bubble node.
"""
const Tri7 = Triangle  # Same topology! Node count from basis.
