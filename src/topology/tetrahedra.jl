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
    reference_coordinates(::Tetrahedron{4}) -> SVector{4, Vec{3,Float64}}

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
    return SVector(Vec{3,Float64}.((
        (0.0, 0.0, 0.0),  # N1: Corner at origin
        (1.0, 0.0, 0.0),  # N2: Corner along ξ
        (0.0, 1.0, 0.0),  # N3: Corner along η
        (0.0, 0.0, 1.0)   # N4: Corner along ζ
    )))
end

"""
    reference_coordinates(::Tetrahedron{10}) -> SVector{10, Vec{3,Float64}}

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
    return SVector(Vec{3,Float64}.((
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
    )))
end

# ============================================================================
# TOPOLOGICAL CONNECTIVITY (Typed Entities)
# ============================================================================

"""
    edges(::T) where T <: Tetrahedron -> SVector{6, Edge{T}}

Return typed edge entities for tetrahedron.

Returns an `SVector` of `Edge{T}` instances. Position in the vector IS the edge ID.

# Returns
6 edges, each containing the vertex indices that bound the edge:
- Edge 1: vertices (1, 2)
- Edge 2: vertices (2, 3)
- Edge 3: vertices (3, 1)
- Edge 4: vertices (1, 4)
- Edge 5: vertices (2, 4)
- Edge 6: vertices (3, 4)

# Examples
```julia
edges_list = edges(Tet4())
# edges_list[1] is Edge 1, bounded by vertices (1,2)
# edges_list[3] is Edge 3, bounded by vertices (3,1)

# Extract from DOF type
DOF{Vec{3}, Edge{Tet4}}
entity_list = entities(Edge{Tet4})  # Type carries all info!
```

# Note
Same for all tetrahedron types (Tet4, Tet10) - topologically identical.
"""
function edges(::T) where {T<:Tetrahedron}
    return SVector(Edge{T}.((
        (1, 2),
        (2, 3),
        (3, 1),
        (1, 4),
        (2, 4),
        (3, 4)
    )))
end

"""
    faces(::T) where T <: Tetrahedron -> SVector{4, Face{T}}

Return typed face entities for tetrahedron.

Returns an `SVector` of `Face{T}` instances. Position in the vector IS the face ID.

# Returns
4 triangular faces, each containing the vertex indices that bound the face:
- Face 1: vertices (1, 3, 2)
- Face 2: vertices (1, 2, 4)
- Face 3: vertices (2, 3, 4)
- Face 4: vertices (3, 1, 4)

# Examples
```julia
faces_list = faces(Tet4())
# faces_list[1] is Face 1, bounded by vertices (1,3,2)
# faces_list[4] is Face 4, bounded by vertices (3,1,4)

# Extract from DOF type
DOF{Vec{3}, Face{Tet4}}
entity_list = entities(Face{Tet4})  # Type carries all info!
```

# Note
Same for all tetrahedron types (Tet4, Tet10) - topologically identical.
"""
function faces(::T) where {T<:Tetrahedron}
    return SVector(Face{T}.((
        (1, 3, 2),
        (1, 2, 4),
        (2, 3, 4),
        (3, 1, 4)
    )))
end

"""
    vertices(::T) where T <: Tetrahedron -> SVector{4, Vertex{T}}

Return typed vertex entities for tetrahedron.

Returns an `SVector` of `Vertex{T}` instances. Position in the vector IS the vertex ID.

# Returns
4 vertices (the corner nodes)

# Examples
```julia
verts = vertices(Tet4())
# verts[1] is Vertex 1
# verts[2] is Vertex 2
# etc.

# Extract from DOF type (Lagrange elements)
DOF{Float64, Vertex{Tet4}}
entity_list = entities(Vertex{Tet4})  # Type carries all info!
```
"""
function vertices(::T) where {T<:Tetrahedron}
    return SVector(Vertex{T}(), Vertex{T}(), Vertex{T}(), Vertex{T}())
end

"""
    cells(::T) where T <: Tetrahedron -> SVector{1, Cell{T}}

Return typed cell entity for tetrahedron (the element interior itself).

Returns an `SVector` with one `Cell{T}` instance (the tetrahedron volume).

# Examples
```julia
cell_list = cells(Tet4())
# cell_list[1] is the Cell (the tetrahedron interior)

# Extract from DOF type (DG elements)
DOF{Float64, Cell{Tet4}}
entity_list = entities(Cell{Tet4})  # Type carries all info!
```
"""
function cells(::T) where {T<:Tetrahedron}
    return SVector(Cell{T}())
end

# ============================================================================
# ENTITY COUNT HELPERS
# ============================================================================

"""
    nvertices(::Tetrahedron) -> Int

Return number of vertices (corner nodes) - always 4 for tetrahedra.
"""
nvertices(::Tetrahedron) = 4

"""
    nedges(::Tetrahedron) -> Int

Return number of edges - always 6 for tetrahedra.
"""
nedges(::Tetrahedron) = 6

"""
    nfaces(::Tetrahedron) -> Int

Return number of faces - always 4 for tetrahedra.
"""
nfaces(::Tetrahedron) = 4

# ============================================================================
# EXPORTS
# ============================================================================

# Export ONLY canonical aliases (not the parametric struct)
export Tet4, Tet10
