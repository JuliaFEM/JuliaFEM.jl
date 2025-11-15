# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE

"""
    Tetrahedron{N} <: AbstractTopology

Parametric tetrahedral element topology (3D simplex).

The type parameter `N` specifies the total number of nodes in the element,
enabling compile-time dispatch and type-stable code generation.

# Type Parameter
- `N::Int`: Total number of nodes (4 or 10)

# Canonical Type Aliases
**Always use these aliases instead of constructing `Tetrahedron{N}` directly:**

- `Tet4 = Tetrahedron{4}` - Linear tetrahedron (P1, 4 corner nodes)
- `Tet10 = Tetrahedron{10}` - Quadratic tetrahedron (P2, 10 nodes: 4 corners + 6 edge midpoints)

# Why Parametric Types?
1. **Type Stability:** Each node count is a distinct type (`Tet4 !== Tet10`)
2. **Compile-Time Dispatch:** Kernel specialization for GPU performance
3. **Zero Allocation:** Node count known at compile time
4. **Clear API:** `nnodes(Tet10())` returns compile-time constant `10`

# Reference Element
```
      N4 (0,0,1)
      /|\\
     / | \\
    /  |  \\
   /   |   \\
  N1---+----N3
  (0,0,0)  (0,1,0)
   \\  /
    \\/
    N2 (1,0,0)
```

# Topology Properties
- Dimension: 3
- Corner nodes: 4
- Edges: 6
- Faces: 4 (triangular)

# Typical Usage
```julia
julia> topology = Tet10()  # Use canonical alias
julia> nnodes(topology)    # Returns compile-time constant
10

julia> Tet4 !== Tet10      # Type stability check
true

julia> reference_coordinates(Tet4())  # Corner nodes only
((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
```

# Design Notes
- Separates topology (geometric shape) from interpolation (basis functions)
- Corner node positions are ALWAYS the same (4 nodes)
- Intermediate nodes (edge midpoints) depend on `N` parameter
- Use `reference_coordinates(Tet10())` to get ALL 10 node positions

# Type Parameter
Node count comes from mesh connectivity. Type parameter enables compile-time optimization.

# Node Count Variants
- `Tetrahedron{4}` (alias `Tet4`): Linear tetrahedron (P1 Lagrange)
- `Tetrahedron{10}` (alias `Tet10`): Quadratic tetrahedron (P2 Lagrange)
"""
struct Tetrahedron{N} <: AbstractTopology{N} end

# ============================================================================
# CANONICAL TYPE ALIASES (PRIMARY API)
# ============================================================================

"""
    Tet4 = Tetrahedron{4}

Linear tetrahedron with 4 corner nodes (P1 interpolation).

**Reference Coordinates:**
- Node 1: (0.0, 0.0, 0.0) - Origin
- Node 2: (1.0, 0.0, 0.0) - Along ξ-axis
- Node 3: (0.0, 1.0, 0.0) - Along η-axis
- Node 4: (0.0, 0.0, 1.0) - Along ζ-axis

**Use this alias everywhere** instead of `Tetrahedron{4}`.
"""
const Tet4 = Tetrahedron{4}

"""
    Tet10 = Tetrahedron{10}

Quadratic tetrahedron with 10 nodes (P2 interpolation).

**Node Layout:**
- Nodes 1-4: Corner nodes (same as Tet4)
- Nodes 5-10: Edge midpoints

**Use this alias everywhere** instead of `Tetrahedron{10}`.
"""
const Tet10 = Tetrahedron{10}

# ============================================================================
# CORE TOPOLOGY INTERFACE
# ============================================================================

"""
    nnodes(::Tetrahedron{N}) where N -> Int

Return total number of nodes for parametric tetrahedron topology.
This is a **compile-time constant** enabling type-stable dispatch.

# Returns
- `N`: Node count specified by type parameter (4 or 10)

# Examples
```julia
julia> nnodes(Tet4())   # Returns compile-time constant 4
4

julia> nnodes(Tet10())  # Returns compile-time constant 10
10

julia> @allocated nnodes(Tet10())  # Zero allocation
0
```

# Performance Note
This function returns a compile-time constant, enabling:
- Zero-cost abstraction (compiler eliminates call)
- Fully specialized code generation
- Static memory allocation in GPU kernels
"""
nnodes(::Tetrahedron{N}) where {N} = N

"""
    dim(::Tetrahedron{N}) where N -> Int

Return spatial dimension of tetrahedron reference element (always 3).

# Returns
- `3`: Tetrahedra exist in 3D space

# Examples
```julia
julia> dim(Tet4())
3

julia> dim(Tet10())  # Same for all tetrahedron types
3
```
"""
dim(::Tetrahedron{N}) where {N} = 3

# ============================================================================
# REFERENCE COORDINATES (Full Node Positions)
# ============================================================================

"""
    reference_coordinates(::Tetrahedron{4}) -> NTuple{4, NTuple{3, Float64}}

Return reference coordinates for linear tetrahedron (Tet4) - 4 corner nodes only.

# Returns
Tuple of 4 coordinate triples: ((ξ₁, η₁, ζ₁), (ξ₂, η₂, ζ₂), (ξ₃, η₃, ζ₃), (ξ₄, η₄, ζ₄))

# Node Positions
```
      N4 (0,0,1)
      /|\\
     / | \\
    /  |  \\
   /   |   \\
  N1---+----N3
  (0,0,0)  (0,1,0)
   \\  /
    \\/
    N2 (1,0,0)
```

- Node 1: (0.0, 0.0, 0.0) - Origin
- Node 2: (1.0, 0.0, 0.0) - Along ξ-axis
- Node 3: (0.0, 1.0, 0.0) - Along η-axis
- Node 4: (0.0, 0.0, 1.0) - Along ζ-axis
"""
function reference_coordinates(::Tetrahedron{4})
    return (
        (0.0, 0.0, 0.0),  # N1: Corner at origin
        (1.0, 0.0, 0.0),  # N2: Corner along ξ
        (0.0, 1.0, 0.0),  # N3: Corner along η
        (0.0, 0.0, 1.0)   # N4: Corner along ζ
    )
end

"""
    reference_coordinates(::Tetrahedron{10}) -> NTuple{10, NTuple{3, Float64}}

Return reference coordinates for quadratic tetrahedron (Tet10) - 10 nodes total.

# Node Layout
- Nodes 1-4: Corner nodes (same as Tet4)
- Node 5: Edge midpoint between N1-N2 (0.5, 0.0, 0.0)
- Node 6: Edge midpoint between N2-N3 (0.5, 0.5, 0.0)
- Node 7: Edge midpoint between N3-N1 (0.0, 0.5, 0.0)
- Node 8: Edge midpoint between N1-N4 (0.0, 0.0, 0.5)
- Node 9: Edge midpoint between N2-N4 (0.5, 0.0, 0.5)
- Node 10: Edge midpoint between N3-N4 (0.0, 0.5, 0.5)
"""
function reference_coordinates(::Tetrahedron{10})
    return (
        (0.0, 0.0, 0.0),  # N1: Corner
        (1.0, 0.0, 0.0),  # N2: Corner
        (0.0, 1.0, 0.0),  # N3: Corner
        (0.0, 0.0, 1.0),  # N4: Corner
        (0.5, 0.0, 0.0),  # N5: Midpoint edge 1-2
        (0.5, 0.5, 0.0),  # N6: Midpoint edge 2-3
        (0.0, 0.5, 0.0),  # N7: Midpoint edge 3-1
        (0.0, 0.0, 0.5),  # N8: Midpoint edge 1-4
        (0.5, 0.0, 0.5),  # N9: Midpoint edge 2-4
        (0.0, 0.5, 0.5)   # N10: Midpoint edge 3-4
    )
end

# ============================================================================
# TOPOLOGICAL CONNECTIVITY (Corner Nodes Only)
# ============================================================================

"""
    edges(::Tetrahedron{N}) where N -> NTuple{6, NTuple{2, Int}}

Return edge connectivity (pairs of **corner node indices**) for tetrahedron.
This is TOPOLOGICAL connectivity, independent of interpolation order.

# Returns
6-tuple of edge definitions:
- Edge 1: (1, 2) - N1 → N2
- Edge 2: (2, 3) - N2 → N3
- Edge 3: (3, 1) - N3 → N1
- Edge 4: (1, 4) - N1 → N4
- Edge 5: (2, 4) - N2 → N4
- Edge 6: (3, 4) - N3 → N4

# Note
- Only references **corner nodes** (1, 2, 3, 4)
- Same for all tetrahedron types (Tet4, Tet10)
"""
function edges(::Tetrahedron{N}) where {N}
    return (
        (1, 2),  # Edge 1
        (2, 3),  # Edge 2
        (3, 1),  # Edge 3
        (1, 4),  # Edge 4
        (2, 4),  # Edge 5
        (3, 4)   # Edge 6
    )
end

"""
    faces(::Tetrahedron{N}) where N -> NTuple{4, NTuple{3, Int}}

Return face connectivity for tetrahedron.
Each face is triangular (3 **corner nodes**).

# Returns
4-tuple of triangular faces:
- Face 1: (1, 3, 2) - Base (looking from above)
- Face 2: (1, 2, 4)
- Face 3: (2, 3, 4)
- Face 4: (3, 1, 4)

# Note
- Only references corner nodes
- Faces are triangular (3 nodes each)
- Same for all tetrahedron types
"""
function faces(::Tetrahedron{N}) where {N}
    return (
        (1, 3, 2),  # Face 1: Base
        (1, 2, 4),  # Face 2
        (2, 3, 4),  # Face 3
        (3, 1, 4)   # Face 4
    )
end

# ============================================================================
# EXPORTS
# ============================================================================

# Export ONLY canonical aliases (not the parametric struct)
export Tet4, Tet10
