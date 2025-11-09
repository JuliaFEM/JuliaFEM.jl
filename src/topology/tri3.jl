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
- `Nedelec{Triangle, 1}` → 3 nodes, but DOFs on edges

# Reference Element
```
     η
     ^
     |
  (0,1)
     |  \\
     |    \\
     |      \\
     +---------> ξ
  (0,0)    (1,0)
```

# Standard Node Positions
For `Lagrange{Triangle, 1}` (P1, 3 nodes):
1. (0, 0) - Origin
2. (1, 0) - Along ξ-axis
3. (0, 1) - Along η-axis

For higher-order bases, additional nodes are added on edges and interior.

# Topology Properties
- Dimension: 2
- Edges: 3
- Faces: 1 (the element itself in 2D)

# Typical Usage
```julia
julia> topology = Triangle()
julia> dim(topology)
2
julia> basis = Lagrange{Triangle, 1}()
julia> nnodes(basis)  # Node count comes from BASIS, not topology
3
```

See also: [`AbstractTopology`](@ref), [`Quadrilateral`](@ref), [`Lagrange`](@ref)
"""
struct Triangle <: AbstractTopology end

dim(::Triangle) = 2

"""
    reference_coordinates(::Triangle)

Get standard reference node positions for Triangle (3 vertices in parametric space).
These are the P1 (linear) node positions by default.
"""
function reference_coordinates(::Triangle)
    return (
        (0.0, 0.0),  # Node 1: Origin
        (1.0, 0.0),  # Node 2: Along ξ-axis
        (0.0, 1.0),  # Node 3: Along η-axis
    )
end

function edges(::Triangle)
    return (
        (1, 2),  # Edge 1: Bottom
        (2, 3),  # Edge 2: Right (hypotenuse)
        (3, 1),  # Edge 3: Left
    )
end

# For 2D elements, faces are the element itself
faces(::Triangle) = ((1, 2, 3),)

# ============================================================================
# Deprecated aliases (for backwards compatibility)
# ============================================================================

"""
    Tri3

**DEPRECATED:** Use `Triangle` + `Lagrange{Triangle, 1}` instead.

The old `Tri3` type conflated topology (triangle) with node count (3).
In the new architecture:
- Topology defines geometric shape only
- Basis functions determine node count

Migration:
```julia
# Old (deprecated)
element = Element(Tri3, (1,2,3))

# New (correct)
element = Element(Triangle(), Lagrange{Triangle, 1}(), Gauss{2}(), (1,2,3))
```
"""
const Tri3 = Triangle

# Note: Tri3 is deprecated. Use Triangle with parametric Lagrange basis instead.

"""
    nnodes(::Triangle) -> Int

Number of corner nodes for a triangle element (3).

**Note:** In the new architecture, actual node count depends on basis degree.
This returns the number of corner nodes for backwards compatibility.
"""
nnodes(::Triangle) = 3
