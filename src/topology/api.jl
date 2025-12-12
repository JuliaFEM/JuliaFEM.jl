# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Topology API definitions.

Defines element topology abstractions - geometric shape and node ordering
of finite elements in their reference configuration.

See `src/topology/README.md` for complete documentation.
"""

# ============================================================================
# TOPOLOGY ABSTRACTIONS
# ============================================================================

"""
    AbstractTopology{N}

Abstract type for element topology (geometric shape and node ordering).

Topology defines the **shape** of an element in reference space: coordinates,
edge/face connectivity, and spatial dimension.

# Type Parameter
- `N::Int`: Number of nodes (from mesh connectivity)

# Interface Requirements
All topology types must implement:
- `nnodes(topology)` - Number of nodes
- `dim(topology)` - Spatial dimension (1, 2, or 3)
- `reference_coordinates(topology)` - Node positions (SVector of Vec)
- `edges(topology)` - Edge connectivity (tuple of tuples)
- `faces(topology)` - Face connectivity (tuple of tuples, 3D only)

# Examples
```julia
Triangle{3} <: AbstractTopology{3}   # 3-node triangle
Triangle{6} <: AbstractTopology{6}   # 6-node triangle
Hexahedron{8} <: AbstractTopology{8}   # 8-node hex
```

See `src/topology/README.md` for comprehensive documentation.
"""
abstract type AbstractTopology{N} end

# ============================================================================
# TOPOLOGY INTERFACE FUNCTIONS
# ============================================================================

"""
    nnodes(topology) -> Int

Number of nodes in the reference element (compile-time constant from type parameter N).
"""
nnodes(::AbstractTopology{N}) where N = N
nnodes(::Type{<:AbstractTopology{N}}) where N = N

"""
    nedges(topology) -> Int

Number of edges in the topology.
"""
nedges(t::AbstractTopology) = length(edges(t))
nedges(::Type{T}) where {T<:AbstractTopology} = length(edges(T()))

"""
    nfaces(topology) -> Int

Number of faces in the topology (3D only).
"""
nfaces(t::AbstractTopology) = length(faces(t))
nfaces(::Type{T}) where {T<:AbstractTopology} = length(faces(T()))

"""
    dim(topology) -> Int

Spatial dimension of the topology (1, 2, or 3).

Each concrete topology must implement this.
"""
function dim end

"""
    Base.ndims(topology) -> Int

Alias to `dim` for Base API compatibility.
"""
Base.ndims(t::AbstractTopology) = dim(t)
Base.ndims(::Type{T}) where {T<:AbstractTopology} = dim(T())

"""
    reference_coordinates(topology) -> SVector{N, Vec{D,Float64}}

Reference element coordinates for the topology's nodes.

Returns `SVector` of `Vec` coordinate vectors (zero allocation).

Each concrete topology must implement this.
"""
function reference_coordinates end

"""
    edges(topology) -> NTuple{M, NTuple{2, Int}}

Edge connectivity (tuple of node index pairs).

Each concrete topology must implement this.
"""
function edges end

"""
    faces(topology) -> NTuple{M, NTuple{K, Int}}

Face connectivity for 3D topologies (tuple of node index tuples).

Each concrete 3D topology must implement this.
"""
function faces end

"""
    cells(topology) -> SVector{M, Cell}

Cell entities for the topology (typically one cell per element).

Each concrete topology must implement this.
"""
function cells end

"""
    vertices(topology) -> SVector{M, Vertex}

Vertex entities for the topology.

Each concrete topology must implement this.
"""
function vertices end

# ============================================================================
# ENTITIES DISPATCHER
# ============================================================================

"""
    entities(::Type{Topo}, ::Val{D}) where {Topo<:AbstractTopology, D}

Return entities of dimension D for the given topology.

Dispatches to dimension-specific functions:
- D=0 → vertices(topology)
- D=1 → edges(topology)
- D=2 → faces(topology)
- D=3 → cells(topology)
"""
entities(::Type{Topo}, ::Val{0}) where {Topo<:AbstractTopology} = vertices(Topo())
entities(::Type{Topo}, ::Val{1}) where {Topo<:AbstractTopology} = edges(Topo())
entities(::Type{Topo}, ::Val{2}) where {Topo<:AbstractTopology} = faces(Topo())
entities(::Type{Topo}, ::Val{3}) where {Topo<:AbstractTopology} = cells(Topo())

# Integer dimension interface
entities(topo::Type{<:AbstractTopology}, d::Int) = entities(topo, Val(d))

# ============================================================================
# TOPOLOGICAL ENTITIES - Typed structures for geometric primitives
# ============================================================================

"""
    TopologicalEntity{D}

Abstract type for topological entities at dimension `D`.

# Type Parameters
- `D::Int`: Geometric dimension (0=vertex, 1=edge, 2=face, 3=cell)

# Concrete Types
- `Vertex`: 0-dimensional point entity
- `Edge`: 1-dimensional line entity (bounded by 2 vertices)
- `Face`: 2-dimensional surface entity (bounded by edges)
- `Cell`: 3-dimensional volume entity (bounded by faces)
"""
abstract type TopologicalEntity{D} end

"""
    Vertex <: TopologicalEntity{0}

A 0-dimensional point entity (vertex/node).
"""
struct Vertex <: TopologicalEntity{0} end

"""
    Edge <: TopologicalEntity{1}

A 1-dimensional line entity bounded by two vertices.

# Fields
- `vertices::NTuple{2, Int}`: Local vertex indices bounding this edge
"""
struct Edge <: TopologicalEntity{1}
    vertices::NTuple{2, Int}
end

"""
    Face <: TopologicalEntity{2}

A 2-dimensional surface entity bounded by edges.

# Fields
- `vertices::NTuple{N, Int}`: Local vertex indices bounding this face
"""
struct Face <: TopologicalEntity{2}
    vertices::NTuple{N, Int} where N
end

"""
    Cell <: TopologicalEntity{3}

A 3-dimensional volume entity (the element interior itself).
"""
struct Cell <: TopologicalEntity{3} end

# ============================================================================
# ENTITY DIMENSION QUERIES
# ============================================================================

"""
    dim(::Type{<:TopologicalEntity{D}}) where D -> Int

Return the geometric dimension of an entity type.
"""
dim(::Type{<:TopologicalEntity{D}}) where {D} = D

# ============================================================================
# HELPER FUNCTIONS FOR ENTITY COUNTS
# ============================================================================

"""
    nentities(::Type{Topo}, ::Type{<:TopologicalEntity{D}}) where {Topo<:AbstractTopology, D}

Return the number of entities of dimension D for the given topology.
"""
function nentities(::Type{Topo}, ::Type{E}) where {Topo<:AbstractTopology, E<:TopologicalEntity}
    D = entity_dim(E)
    return length(entities(Topo, D))
end

# Helper to extract dimension from entity type
entity_dim(::Type{<:Vertex}) = 0
entity_dim(::Type{<:Edge}) = 1  
entity_dim(::Type{<:Face}) = 2
entity_dim(::Type{<:Cell}) = 3
