# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
    AbstractInterfaceMesh

Supertype for meshes that discretise **interfaces between bodies** (contact,
fluid–solid boundaries, enriched reaction surfaces). Distinct from
[`AbstractMesh`](@ref): volumes use [`Mesh`](@ref); interfaces use these types
plus [`InterfaceDOFHandler`](@ref).

Concrete types: [`InterfaceMesh`](@ref).
"""
abstract type AbstractInterfaceMesh end

"""
    InterfaceVolumeCoupling

Maps one interface element (e.g. a [`Seg2`](@ref) segment in [`InterfaceMesh`](@ref))
to contributing **volume** elements on slave and master sides for mortar / FSI /
coupled quadrature.

Indices are **1-based** Julia conventions. `slave_local_face` / `master_local_face`
index into [`faces(::AbstractTopology)`](@ref) on the respective **volume**
topology (`0` = reserved / not yet filled).

`slave_body` / `master_body` identify which volume [`Mesh`](@ref) or region the element
index refers to (caller-defined numbering).
"""
struct InterfaceVolumeCoupling
    slave_body::UInt32
    slave_elem::UInt32
    slave_local_face::UInt8
    master_body::UInt32
    master_elem::UInt32
    master_local_face::UInt8
end

"""
    InterfaceMesh{N, T<:AbstractTopology{N}} <: AbstractInterfaceMesh

Embedded interface grid: nodes in physical ``\\mathbb{R}^3``, connectivity with the
same tuple layout as volume [`Mesh`](@ref).

- `volume_coupling[e]` pairs interface element `e` with slave/master volume data.
- `segment_sets` / `node_sets` mirror volume mesh sets (named groups).

# Example

```julia
# Two segments sharing node 2; coupling placeholders for future mortar fill-in.
nodes = [Vec(0.0, 0.0, 0.0), Vec(0.5, 0.0, 0.0), Vec(1.0, 0.0, 0.0)]
conn = [(UInt32(1), UInt32(2)), (UInt32(2), UInt32(3))]
coup = [
    InterfaceVolumeCoupling(1, 1, UInt8(1), 2, 1, UInt8(1)),
    InterfaceVolumeCoupling(1, 2, UInt8(1), 2, 2, UInt8(1)),
]
im = InterfaceMesh(Seg2, nodes, conn, coup)
```
"""
struct InterfaceMesh{N, T<:AbstractTopology{N}} <: AbstractInterfaceMesh
    nodes::Vector{Vec{3, Float64}}
    connectivity::Vector{NTuple{N, UInt32}}
    volume_coupling::Vector{InterfaceVolumeCoupling}
    segment_sets::Dict{Symbol, Set{UInt32}}
    node_sets::Dict{Symbol, Set{UInt32}}
end

function InterfaceMesh(
    ::Type{T},
    nodes::Vector{Vec{3, Float64}},
    connectivity::Vector{NTuple{N, UInt32}},
    volume_coupling::Vector{InterfaceVolumeCoupling};
    segment_sets::Dict{Symbol, Set{UInt32}} = Dict{Symbol, Set{UInt32}}(),
    node_sets::Dict{Symbol, Set{UInt32}} = Dict{Symbol, Set{UInt32}}(),
) where {N, T<:AbstractTopology{N}}
    expected = nnodes(T())
    @assert N == expected "InterfaceMesh: tuple size $N must match nnodes($T) = $expected"
    n_nodes = length(nodes)
    for (ei, elem_conn) in enumerate(connectivity)
        for nid in elem_conn
            @assert 1 ≤ nid ≤ n_nodes "InterfaceMesh element $ei: node $nid out of range"
        end
    end
    ne = length(connectivity)
    length(volume_coupling) == ne || throw(
        ArgumentError(
            "InterfaceMesh: length(volume_coupling)=$(length(volume_coupling)) must match nelements=$ne",
        ),
    )
    return InterfaceMesh{N, T}(nodes, connectivity, volume_coupling, segment_sets, node_sets)
end

"""Topology type parameter of `InterfaceMesh{N,T}` (type domain)."""
topology_type(::InterfaceMesh{N, T}) where {N, T} = T

"""Number of nodes on the interface mesh."""
interface_nnodes(im::InterfaceMesh) = length(im.nodes)

"""Number of interface elements (segments, patches, …)."""
interface_nelements(im::InterfaceMesh) = length(im.connectivity)

# Used by InterfaceDOFHandler entity counting (same helper contract as volume DOFHandler).
function _field_entity_count(mesh::InterfaceMesh, ::Type{Vertex}, _)
    return length(mesh.nodes)
end

function _field_entity_count(mesh::InterfaceMesh, ::Type{Cell}, _)
    return length(mesh.connectivity)
end

function _field_entity_count(::InterfaceMesh, ::Type{Edge}, _)
    error("InterfaceMesh: Edge fields require interface facet maps (not implemented).")
end

function _field_entity_count(::InterfaceMesh, ::Type{Face}, _)
    error("InterfaceMesh: Face fields require interface facet maps (not implemented).")
end
