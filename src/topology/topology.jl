# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Topology module - topological entities and helpers.

Abstract type and interface are defined in topology/api.jl.
This file defines the typed entity system for vertices, edges, faces, and cells.
"""

# NOTE: AbstractTopology{N} and interface functions (nnodes, dim, reference_coordinates, edges, faces)
# are now defined in topology/api.jl, which is included before this file in JuliaFEM.jl

# ============================================================================
# TOPOLOGICAL ENTITIES - Typed structures for geometric primitives
# ============================================================================

"""
    TopologicalEntity{D, Topo}

Abstract type for topological entities at dimension `D` belonging to topology `Topo`.

# Type Parameters
- `D::Int`: Geometric dimension (0=vertex, 1=edge, 2=face, 3=cell)
- `Topo <: AbstractTopology`: The topology type this entity belongs to

# Concrete Types
- `Vertex{Topo}`: 0-dimensional point entity
- `Edge{Topo}`: 1-dimensional line entity (bounded by 2 vertices)
- `Face{Topo}`: 2-dimensional surface entity (bounded by edges)
- `Cell{Topo}`: 3-dimensional volume entity (bounded by faces)

# Philosophy

Entities are **topological** (connectivity) not **geometric** (coordinates).
They define "what connects to what" independent of "where things are".

# Usage with DOF System

The entity type encodes WHERE degrees of freedom live:

```julia
# Lagrange elements: DOFs on vertices
DOF{Float64, Vertex{Tet4}}

# Nedelec elements: DOFs on edges  
DOF{Vec{3}, Edge{Tet4}}

# Raviart-Thomas elements: DOFs on faces
DOF{Vec{3}, Face{Tet4}}

# Discontinuous Galerkin: DOFs in cell interior
DOF{Float64, Cell{Tet4}}
```

The type parameter carries complete compile-time information:
- Quantity type (Float64, Vec{3}, etc.)
- Location (Vertex, Edge, Face, Cell)
- Topology (which element type)

# Entity Position as ID

Entities do NOT carry an explicit `id` field. Instead, position in the
returned vector IS the entity ID:

```julia
edge_list = entities(Edge{Tet4})
# edge_list[1] is Edge 1
# edge_list[2] is Edge 2
# etc.
```

This enables zero-allocation, type-stable queries.
"""
abstract type TopologicalEntity{D, Topo <: AbstractTopology} end

"""
    Vertex{Topo} <: TopologicalEntity{0, Topo}

A 0-dimensional point entity (vertex/node).

Vertices are the corner points of an element. Position in the vertex
list IS the vertex ID (no explicit id field needed).

# Examples
```julia
vertex_list = vertices(Tet4())
# vertex_list[1] is Vertex 1 (at local index 1)
# vertex_list[2] is Vertex 2 (at local index 2)
# etc.

# Or use generic interface
vertices = entities(Vertex{Tet4})
```
"""
struct Vertex{Topo} <: TopologicalEntity{0, Topo} end

"""
    Edge{Topo} <: TopologicalEntity{1, Topo}

A 1-dimensional line entity bounded by two vertices.

Edges connect pairs of vertices. Position in the edge list IS the edge ID.

# Fields
- `vertices::NTuple{2, Int}`: Local vertex indices bounding this edge

# Examples
```julia
edge_list = edges(Tet4())
# edge_list[1] is Edge 1, connects vertices edge_list[1].vertices
# edge_list[3] is Edge 3, connects vertices edge_list[3].vertices

# Or use generic interface
edges = entities(Edge{Tet4})

# Usage in DOF Systems
```julia
# Nedelec edge elements
DOF{Vec{3}, Edge{Tet4}}  # Vector DOF on each edge of Tet4
```
"""
struct Edge{Topo} <: TopologicalEntity{1, Topo}
    vertices::NTuple{2, Int}
end

"""
    Face{Topo} <: TopologicalEntity{2, Topo}

A 2-dimensional surface entity bounded by edges.

Faces are surfaces (triangles, quadrilaterals) that bound a volume.
Position in the face list IS the face ID.

# Fields
- `vertices::NTuple{N, Int}`: Local vertex indices bounding this face (N=3 for triangle, N=4 for quad)

# Examples
```julia
face_list = faces(Tet4())
# face_list[1] is Face 1, vertices at face_list[1].vertices
# face_list[2] is Face 2, vertices at face_list[2].vertices

# Or use generic interface
faces = entities(Face{Tet4})

# Usage in DOF Systems
```julia
# Raviart-Thomas face elements
DOF{Vec{3}, Face{Tet4}}  # Vector DOF on each face of Tet4
```
"""
struct Face{Topo} <: TopologicalEntity{2, Topo}
    vertices::NTuple{N, Int} where N
end

"""
    Cell{Topo} <: TopologicalEntity{3, Topo}

A 3-dimensional volume entity (the element interior itself).

For most elements, there is exactly one cell - the element itself.
Position in the cell list IS the cell ID (typically just one cell).

# Examples
```julia
cell_list = cells(Tet4())
# cell_list[1] is the Cell (the tetrahedron interior)

# Or use generic interface
cells = entities(Cell{Tet4})

# Usage in DOF Systems
```julia
# Discontinuous Galerkin elements
DOF{Float64, Cell{Tet4}}  # Scalar DOF in element interior
```
"""
struct Cell{Topo} <: TopologicalEntity{3, Topo} end

# ============================================================================
# ENTITY DIMENSION QUERIES
# ============================================================================

"""
    dim(::Type{<:TopologicalEntity{D}}) where D -> Int

Return the geometric dimension of an entity type.

# Examples
```julia
dim(Vertex{Tet4})  # 0
dim(Edge{Tet4})    # 1
dim(Face{Tet4})    # 2
dim(Cell{Tet4})    # 3
```
"""
dim(::Type{<:TopologicalEntity{D}}) where {D} = D

# ============================================================================
# ENTITY QUERIES - Type-based dispatch
# ============================================================================

"""
    topology_type(::Type{<:TopologicalEntity{D, Topo}}) where {D, Topo}

Extract the topology type from an entity type.

# Examples
```julia
topology_type(Edge{Tet4})    # Tet4
topology_type(Face{Tet10})   # Tet10
topology_type(Vertex{Hex8})  # Hex8
```
"""
topology_type(::Type{<:TopologicalEntity{D, Topo}}) where {D, Topo} = Topo

"""
    entities(::Type{Entity}) where Entity <: TopologicalEntity

Return a vector of all entities of the given type.

Topology is extracted from the entity type parameter - no need to pass it separately!

# Arguments
- `Entity`: Entity type (e.g., `Edge{Tet4}`, `Face{Tet4}`)

# Returns
`SVector` of entity instances. Position in vector IS the entity ID.

# Examples
```julia
# Direct entity queries (topology embedded in type)
vertices = entities(Vertex{Tet4})  # 4 vertices
edges = entities(Edge{Tet4})       # 6 edges
faces = entities(Face{Tet4})       # 4 faces
cells = entities(Cell{Tet4})       # 1 cell

# Extract from DOF type
dof_type = DOF{Vec{3}, Edge{Tet4}}
entity_type = typeof(dof_type).parameters[2]  # Edge{Tet4}
edges = entities(entity_type)  # Type carries all info!
```

# Design Philosophy

The entity type `Edge{Tet4}` already contains the topology `Tet4` as a type parameter.
No need to pass topology separately - just extract it from the type!

```julia
entities(Edge{Tet4})   # Type carries all information
entities(Face{Hex8})   # Clean and concise
```

# Implementation Note

Each topology type must provide `vertices()`, `edges()`, `faces()`, and optionally
`cells()` methods that return `SVector` of the corresponding entity types.
The generic `entities()` dispatcher extracts the topology and routes to these methods.
"""
function entities end

# Extract topology from entity type and dispatch
entities(::Type{Vertex{T}}) where {T<:AbstractTopology} = vertices(T())
entities(::Type{Edge{T}}) where {T<:AbstractTopology} = edges(T())
entities(::Type{Face{T}}) where {T<:AbstractTopology} = faces(T())
entities(::Type{Cell{T}}) where {T<:AbstractTopology} = cells(T())

# ============================================================================
# HELPER FUNCTIONS FOR ENTITY COUNTS
# ============================================================================

"""
    nentities(::Type{Entity}) where Entity <: TopologicalEntity

Return the number of entities of the given type.

# Examples
```julia
nentities(Vertex{Tet4})  # 4
nentities(Edge{Tet4})    # 6
nentities(Face{Tet4})    # 4
nentities(Cell{Tet4})    # 1
```
"""
nentities(entity_type::Type{<:TopologicalEntity}) = length(entities(entity_type))
