# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Topology API definitions.

This file defines element topology abstractions - the geometric shape and node ordering
of finite elements in their reference configuration.

Must be included after core api.jl.
"""

# ============================================================================
# TOPOLOGY ABSTRACTIONS
# ============================================================================

"""
    AbstractTopology

Abstract type for element topology (geometric shape and node ordering).

Topology defines the **shape** of an element in its reference (parametric) space:
- Number of nodes
- Reference element coordinates (ξ, η, ζ positions)
- Edge connectivity (which nodes form edges)
- Face connectivity (which nodes form faces, 3D only)
- Spatial dimension (1D, 2D, or 3D)

# Interface Requirements

All topology types must implement:
- `nnodes(topology)` - Number of nodes
- `dim(topology)` - Spatial dimension (1, 2, or 3)
- `reference_coordinates(topology)` - Node positions in reference element (returns tuple of tuples)
- `edges(topology)` - Edge connectivity (returns tuple of tuples)
- `faces(topology)` - Face connectivity (returns tuple of tuples, 3D only)

# Concrete Types

**1D (Lines):**
- `Segment` - Generic 1D line segment

**2D (Surfaces):**
- `Triangle` - 2D simplex (straight or curved edges)
- `Quadrilateral` - 2D quadrilateral (straight or curved edges)

**3D (Volumes):**
- `Tetrahedron` - 3D simplex (straight or curved faces)
- `Hexahedron` - 3D brick (straight or curved faces)
- `Pyramid` - 3D pyramid (quad base, triangular sides)
- `Wedge` - 3D prism (triangular extrusion)

# Design Philosophy

**Key Insight:** Topology defines SHAPE, not node count!

Node count comes from the **basis function order**:
- Linear basis (P=1): Corner nodes only
- Quadratic basis (P=2): Corner + mid-edge nodes
- Cubic basis (P=3): Corner + edge + face nodes

**Examples:**
```julia
# Triangle with different basis orders
Triangle + Lagrange{Triangle, 1} → 3 nodes  (corners)
Triangle + Lagrange{Triangle, 2} → 6 nodes  (corners + mid-edges)
Triangle + Lagrange{Triangle, 3} → 10 nodes (corners + edges + interior)

# Quadrilateral with different basis types
Quadrilateral + Lagrange{Quadrilateral, 1}     → 4 nodes  (corners)
Quadrilateral + Serendipity{Quadrilateral, 2} → 8 nodes  (corners + mid-edges, no center)
Quadrilateral + Lagrange{Quadrilateral, 2}     → 9 nodes  (corners + mid-edges + center)
```

**Separation of Concerns:**
- Topology: "This is a triangle" (shape)
- Basis: "This triangle has 6 nodes" (interpolation)
- Integration: "Use 3-point Gauss rule" (numerical quadrature)

# Backward Compatibility

Old names like `Tri3`, `Quad4`, `Tet10` are **deprecated** but aliased:
- `Tri3` → `Triangle` (with implied linear basis)
- `Quad4` → `Quadrilateral` (with implied linear basis)
- `Tet10` → `Tetrahedron` (with implied quadratic basis)

New code should use shape names (`Triangle`) with explicit basis specification.

# Reference Element Coordinates

Each topology has standard reference coordinates:

**Segment:** ξ ∈ [-1, 1]
**Triangle:** (ξ, η) where ξ, η ≥ 0 and ξ + η ≤ 1
**Quadrilateral:** (ξ, η) ∈ [-1, 1] × [-1, 1]
**Tetrahedron:** (ξ, η, ζ) where ξ, η, ζ ≥ 0 and ξ + η + ζ ≤ 1
**Hexahedron:** (ξ, η, ζ) ∈ [-1, 1]³
**Pyramid:** (ξ, η, ζ) where (ξ, η) ∈ [-1, 1]² and ζ ∈ [0, 1], with ξ²+η² ≤ (1-ζ)²
**Wedge:** (ξ, η, ζ) where (ξ, η) triangle and ζ ∈ [-1, 1]

# Usage

```julia
# Query topology properties
topology = Triangle()
dim(topology)                    # 2
reference_coordinates(topology)  # ((0,0), (1,0), (0,1))
edges(topology)                  # ((1,2), (2,3), (3,1))

# Topology is independent of basis order
element_linear = Element(Triangle, Lagrange{Triangle,1}, (1,2,3))    # 3 nodes
element_quad   = Element(Triangle, Lagrange{Triangle,2}, (1,2,3,4,5,6))  # 6 nodes

# Both elements have the same topology (Triangle), different basis orders
```

# See Also
- [`dim`](@ref) - Spatial dimension
- [`nnodes`](@ref) - Number of nodes (depends on basis, not topology!)
- [`reference_coordinates`](@ref) - Reference element node positions
- [`edges`](@ref) - Edge connectivity
- [`faces`](@ref) - Face connectivity (3D only)
- Architecture docs: `docs/book/element_architecture.md`

# Type Parameter

`AbstractTopology{N}` where `N` is the number of nodes. Node count comes from mesh connectivity.

# Examples
```julia
Triangle{3} <: AbstractTopology{3}   # 3-node triangle (linear)
Triangle{6} <: AbstractTopology{6}   # 6-node triangle (quadratic)
Hexahedron{8} <: AbstractTopology{8}   # 8-node hex (linear)
Hexahedron{20} <: AbstractTopology{20} # 20-node hex (quadratic serendipity)
Hexahedron{27} <: AbstractTopology{27} # 27-node hex (quadratic full)
```

# Rationale

Node count is included in the type parameter for compile-time performance optimization:
- Enables `Val(N)` for zero-allocation ntuple operations
- Allows loop unrolling for small N
- Node count comes from mesh connectivity, not basis choice
- See ADR-002 for detailed design rationale
"""
abstract type AbstractTopology{N} end

# ============================================================================
# TOPOLOGY INTERFACE FUNCTIONS
# ============================================================================

"""
    nnodes(topology::AbstractTopology{N}) -> Int

Number of nodes in the reference element.

This is a compile-time constant derived from the type parameter `N`.

# Examples

```julia
nnodes(Triangle{3}())   # 3
nnodes(Triangle{6}())   # 6
nnodes(Quadrilateral{4}())  # 4
nnodes(Quadrilateral{9}())  # 9
nnodes(Hexahedron{8}())     # 8
nnodes(Hexahedron{27}())    # 27
```

# Implementation

The default implementation extracts `N` from the type parameter:
```julia
nnodes(::AbstractTopology{N}) where N = N
```

Concrete types inherit this implementation automatically.
"""
nnodes(::AbstractTopology{N}) where N = N

"""
    dim(topology::AbstractTopology) -> Int

Spatial dimension of the topology (1, 2, or 3).

# Examples

```julia
dim(Segment())       # 1
dim(Triangle())      # 2
dim(Quadrilateral()) # 2
dim(Tetrahedron())   # 3
dim(Hexahedron())    # 3
dim(Pyramid())       # 3
dim(Wedge())         # 3
```

# Implementation

Each concrete topology type must provide:
```julia
dim(::Segment) = 1
dim(::Triangle) = 2
dim(::Tetrahedron) = 3
# etc.
```
"""
function dim end

"""
    reference_coordinates(topology::AbstractTopology) -> NTuple{N, NTuple{D, Float64}}

Reference element coordinates for the topology's nodes.

Returns a tuple of coordinate tuples, one per node.
The actual number of nodes depends on the basis order (not shown here).

# Examples

```julia
# Triangle (linear: 3 nodes, quadratic: 6 nodes, etc.)
reference_coordinates(Triangle())
# For linear basis (3 nodes):
# ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))

# Quadrilateral (4 corner nodes minimum)
reference_coordinates(Quadrilateral())
# ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
```

# Note

This returns coordinates for **corner nodes** by default.
Mid-edge and interior nodes are computed by the basis function module.

# Implementation

Each concrete topology type must provide:
```julia
reference_coordinates(::Triangle) = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
reference_coordinates(::Quadrilateral) = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
# etc.
```
"""
function reference_coordinates end

"""
    edges(topology::AbstractTopology) -> NTuple{N, NTuple{2, Int}}

Edge connectivity for the topology.

Returns a tuple of 2-tuples, each containing node indices that form an edge.

# Examples

```julia
# Triangle has 3 edges
edges(Triangle())  # ((1,2), (2,3), (3,1))

# Quadrilateral has 4 edges
edges(Quadrilateral())  # ((1,2), (2,3), (3,4), (4,1))

# Tetrahedron has 6 edges
edges(Tetrahedron())  # ((1,2), (2,3), (3,1), (1,4), (2,4), (3,4))
```

# Usage

Edge connectivity is used for:
- Surface extraction
- Boundary condition application
- Contact surface identification
- Mesh refinement (edge splitting)

# Implementation

Each concrete topology type must provide:
```julia
edges(::Triangle) = ((1,2), (2,3), (3,1))
edges(::Quadrilateral) = ((1,2), (2,3), (3,4), (4,1))
# etc.
```
"""
function edges end

"""
    faces(topology::AbstractTopology) -> NTuple{N, NTuple{M, Int}}

Face connectivity for 3D topologies.

Returns a tuple of tuples, each containing node indices that form a face.
Only applicable to 3D topologies (Tetrahedron, Hexahedron, Pyramid, Wedge).

# Examples

```julia
# Tetrahedron has 4 triangular faces
faces(Tetrahedron())
# ((1,3,2), (1,2,4), (1,4,3), (2,3,4))

# Hexahedron has 6 quadrilateral faces
faces(Hexahedron())
# ((1,4,3,2), (1,2,6,5), (2,3,7,6), (3,4,8,7), (4,1,5,8), (5,6,7,8))

# Pyramid has 1 quad base + 4 triangular sides
faces(Pyramid())
# ((1,4,3,2), (1,2,5), (2,3,5), (3,4,5), (4,1,5))
```

# Usage

Face connectivity is used for:
- Surface element creation
- Traction boundary conditions
- Contact surface identification
- Visualization
- Mesh refinement (face splitting)

# Note

2D topologies do not have faces (they ARE faces).
Calling `faces()` on 2D topology should error or return empty tuple.

# Implementation

Each concrete 3D topology type must provide:
```julia
faces(::Tetrahedron) = ((1,3,2), (1,2,4), (1,4,3), (2,3,4))
faces(::Hexahedron) = ((1,4,3,2), (1,2,6,5), (2,3,7,6), (3,4,8,7), (4,1,5,8), (5,6,7,8))
# etc.
```
"""
function faces end
