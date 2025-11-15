# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
    Triangle{N} <: AbstractTopology

Parametric triangular element topology in 2D.

The type parameter `N` specifies the total number of nodes in the element,
enabling compile-time dispatch and type-stable code generation.

# Type Parameter
- `N::Int`: Total number of nodes (3, 6, 7, or 10)

# Canonical Type Aliases
**Always use these aliases instead of constructing `Triangle{N}` directly:**

- `Tri3 = Triangle{3}` - Linear triangle (P1, 3 corner nodes)
- `Tri6 = Triangle{6}` - Quadratic triangle (P2, 6 nodes: 3 corners + 3 edge midpoints)
- `Tri7 = Triangle{7}` - Quadratic triangle with centroid (3 corners + 3 edge midpoints + 1 center)
- `Tri10 = Triangle{10}` - Cubic triangle (P3, 10 nodes)

# Why Parametric Types?
1. **Type Stability:** Each node count is a distinct type (`Tri3 !== Tri6`)
2. **Compile-Time Dispatch:** Kernel specialization for GPU performance
3. **Zero Allocation:** Node count known at compile time
4. **Clear API:** `nnodes(Tri6())` returns compile-time constant `6`

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

# Topology Properties
- Dimension: 2
- Corner nodes: 3
- Edges: 3
- Faces: 1 (the element itself in 2D)

# Typical Usage
```julia
julia> topology = Tri6()  # Use canonical alias
julia> nnodes(topology)   # Returns compile-time constant
6

julia> Tri3 !== Tri6      # Type stability check
true

julia> reference_coordinates(Tri3())  # Corner nodes only
((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
```

# Design Notes
- Separates topology (geometric shape) from interpolation (basis functions)
- Corner node positions are ALWAYS the same (3 nodes)
- Intermediate nodes (edge/face) depend on `N` parameter
- Use `reference_coordinates(Tri6())` to get ALL 6 node positions
"""
struct Triangle{N} <: AbstractTopology{N} end

# ============================================================================
# CANONICAL TYPE ALIASES (PRIMARY API)
# ============================================================================

"""
    Tri3 = Triangle{3}

Linear triangle with 3 corner nodes (P1 interpolation).

**Reference Coordinates:**
- Node 1: (0.0, 0.0) - Origin
- Node 2: (1.0, 0.0) - Along ξ-axis
- Node 3: (0.0, 1.0) - Along η-axis

**Use this alias everywhere** instead of `Triangle{3}`.
"""
const Tri3 = Triangle{3}

"""
    Tri6 = Triangle{6}

Quadratic triangle with 6 nodes (P2 interpolation).

**Node Layout:**
- Nodes 1-3: Corner nodes (same as Tri3)
- Nodes 4-6: Edge midpoints

**Use this alias everywhere** instead of `Triangle{6}`.
"""
const Tri6 = Triangle{6}

"""
    Tri7 = Triangle{7}

Quadratic triangle with 7 nodes (includes face centroid).

**Node Layout:**
- Nodes 1-3: Corner nodes
- Nodes 4-6: Edge midpoints
- Node 7: Face centroid

**Use this alias everywhere** instead of `Triangle{7}`.
"""
const Tri7 = Triangle{7}

"""
    Tri10 = Triangle{10}

Cubic triangle with 10 nodes (P3 interpolation).

**Node Layout:**
- Nodes 1-3: Corner nodes
- Nodes 4-9: Two nodes per edge (at 1/3 and 2/3 positions)
- Node 10: Face centroid

**Use this alias everywhere** instead of `Triangle{10}`.
"""
const Tri10 = Triangle{10}

# ============================================================================
# CORE TOPOLOGY INTERFACE
# ============================================================================

"""
    nnodes(::Triangle{N}) where N -> Int

Return total number of nodes for parametric triangle topology.
This is a **compile-time constant** enabling type-stable dispatch.

# Returns
- `N`: Node count specified by type parameter (3, 6, 7, or 10)

# Examples
```julia
julia> nnodes(Tri3())   # Returns compile-time constant 3
3

julia> nnodes(Tri6())   # Returns compile-time constant 6
6

julia> @allocated nnodes(Tri6())  # Zero allocation
0
```

# Performance Note

This function returns a compile-time constant, enabling:
- Zero-cost abstraction (compiler eliminates call)
- Fully specialized code generation
- Static memory allocation in GPU kernels
"""
nnodes(::Triangle{N}) where {N} = N

"""
    dim(::Triangle{N}) where N -> Int

Return spatial dimension of triangle reference element (always 2).

# Returns
- `2`: Triangles exist in 2D space

# Examples
```julia
julia> dim(Tri3())
2

julia> dim(Tri10())  # Same for all triangle types
2
```
"""
dim(::Triangle{N}) where {N} = 2

# ============================================================================
# REFERENCE COORDINATES (Full Node Positions)
# ============================================================================

"""
    reference_coordinates(::Triangle{3}) -> NTuple{3, NTuple{2, Float64}}

Return reference coordinates for linear triangle (Tri3) - 3 corner nodes only.

# Returns
Tuple of 3 coordinate pairs: ((ξ₁, η₁), (ξ₂, η₂), (ξ₃, η₃))

# Node Positions
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

- Node 1: (0.0, 0.0) - Origin
- Node 2: (1.0, 0.0) - Along ξ-axis
- Node 3: (0.0, 1.0) - Along η-axis
"""
function reference_coordinates(::Triangle{3})
    return (
        (0.0, 0.0),  # N1: Corner at origin
        (1.0, 0.0),  # N2: Corner along ξ
        (0.0, 1.0)   # N3: Corner along η
    )
end

"""
    reference_coordinates(::Triangle{6}) -> NTuple{6, NTuple{2, Float64}}

Return reference coordinates for quadratic triangle (Tri6) - 6 nodes total.

# Node Layout

- Nodes 1-3: Corner nodes (same as Tri3)
- Node 4: Edge midpoint between N1-N2 (0.5, 0.0)
- Node 5: Edge midpoint between N2-N3 (0.5, 0.5)
- Node 6: Edge midpoint between N3-N1 (0.0, 0.5)
"""
function reference_coordinates(::Triangle{6})
    return (
        (0.0, 0.0),  # N1: Corner
        (1.0, 0.0),  # N2: Corner
        (0.0, 1.0),  # N3: Corner
        (0.5, 0.0),  # N4: Midpoint of edge 1-2
        (0.5, 0.5),  # N5: Midpoint of edge 2-3
        (0.0, 0.5)   # N6: Midpoint of edge 3-1
    )
end

"""
    reference_coordinates(::Triangle{7}) -> NTuple{7, NTuple{2, Float64}}

Return reference coordinates for quadratic triangle with centroid (Tri7).

# Node Layout

- Nodes 1-3: Corner nodes
- Nodes 4-6: Edge midpoints
- Node 7: Face centroid (1/3, 1/3)
"""
function reference_coordinates(::Triangle{7})
    return (
        (0.0, 0.0),          # N1: Corner
        (1.0, 0.0),          # N2: Corner
        (0.0, 1.0),          # N3: Corner
        (0.5, 0.0),          # N4: Midpoint edge 1-2
        (0.5, 0.5),          # N5: Midpoint edge 2-3
        (0.0, 0.5),          # N6: Midpoint edge 3-1
        (1.0 / 3.0, 1.0 / 3.0)   # N7: Face centroid
    )
end

"""
    reference_coordinates(::Triangle{10}) -> NTuple{10, NTuple{2, Float64}}

Return reference coordinates for cubic triangle (Tri10) - 10 nodes total.

# Node Layout

- Nodes 1-3: Corner nodes
- Nodes 4-9: Two nodes per edge (at 1/3 and 2/3)
  - Edge 1-2: N4 (1/3, 0), N5 (2/3, 0)
  - Edge 2-3: N6 (2/3, 1/3), N7 (1/3, 2/3)
  - Edge 3-1: N8 (0, 2/3), N9 (0, 1/3)
- Node 10: Face centroid (1/3, 1/3)
"""
function reference_coordinates(::Triangle{10})
    return (
        (0.0, 0.0),          # N1: Corner
        (1.0, 0.0),          # N2: Corner
        (0.0, 1.0),          # N3: Corner
        (1.0 / 3.0, 0.0),      # N4: Edge 1-2, 1/3
        (2.0 / 3.0, 0.0),      # N5: Edge 1-2, 2/3
        (2.0 / 3.0, 1.0 / 3.0),  # N6: Edge 2-3, 1/3
        (1.0 / 3.0, 2.0 / 3.0),  # N7: Edge 2-3, 2/3
        (0.0, 2.0 / 3.0),      # N8: Edge 3-1, 1/3
        (0.0, 1.0 / 3.0),      # N9: Edge 3-1, 2/3
        (1.0 / 3.0, 1.0 / 3.0)   # N10: Face centroid
    )
end

# ============================================================================
# TOPOLOGICAL CONNECTIVITY (Corner Nodes Only)
# ============================================================================

"""
    edges(::Triangle{N}) where N -> NTuple{3, NTuple{2, Int}}

Return edge connectivity (pairs of **corner node indices**) for triangle.
This is TOPOLOGICAL connectivity, independent of interpolation order.

# Returns

3-tuple of edge definitions:
- Edge 1: (1, 2) - Bottom edge (N1 → N2)
- Edge 2: (2, 3) - Diagonal edge (N2 → N3)
- Edge 3: (3, 1) - Left edge (N3 → N1)

# Note

- Only references **corner nodes** (1, 2, 3)
- Direction: Counter-clockwise
- Same for all triangle types (Tri3, Tri6, Tri7, Tri10)
"""
function edges(::Triangle{N}) where {N}
    return (
        (1, 2),  # Edge 1: Bottom
        (2, 3),  # Edge 2: Diagonal
        (3, 1)   # Edge 3: Left
    )
end

"""
    faces(::Triangle{N}) where N -> NTuple{1, NTuple{3, Int}}

Return face connectivity for triangle.
In 2D, the "face" is the element itself (all 3 **corner nodes**).

# Returns
1-tuple containing the triangular face: ((1, 2, 3),)

# Note
- Only references corner nodes
- Single face represents entire surface
- API consistency with 3D elements
"""
function faces(::Triangle{N}) where {N}
    return ((1, 2, 3),)
end

# ============================================================================
# EXPORTS
# ============================================================================

# Export ONLY canonical aliases (not the parametric struct)
export Tri3, Tri6, Tri7, Tri10
