# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# Mesh-partition adjacency and halo-exchange plumbing.
#
# Despite the filename, this file holds the partition / halo-exchange
# plumbing in full: [`PartitionAdjacency`](@ref) +
# [`build_partition_adjacency`](@ref) describe which partition pairs
# share at least one mesh node, and [`RankHaloExchange`](@ref) carries
# the precomputed sparse send/recv tables that the matrix-free `apply_K!`
# / `apply_M!` paths use to exchange halo DOF data each application.

"""
    PartitionAdjacency(max_part_id, neighbors)

`neighbors[p]::Vector{Int}` lists partition ids sharing at least one mesh node
with partition `p` (undirected graph). Indexed by `p = 1:max_part_id`.

Built by [`build_partition_adjacency`](@ref). Setup allocates; hot paths use
precomputed [`RankHaloExchange`](@ref) data instead.
"""
struct PartitionAdjacency
    max_part_id::Int
    neighbors::Vector{Vector{Int}}
end

"""
    build_partition_adjacency(layout::MeshPartitionLayout, mesh::Mesh) -> PartitionAdjacency

Scan element-node incidence via `mesh.inverse_connectivity`: two partitions
are adjacent if some node carries incident elements from both.
"""
function build_partition_adjacency(
    layout::MeshPartitionLayout,
    mesh::Mesh{N,T},
) where {N,T}
    validate_partition(layout, length(mesh.connectivity))
    np = maximum(layout.element_part_id)
    sets = [Set{Int}() for _ in 1:np]
    @inbounds for e in eachindex(mesh.connectivity)
        pe = layout.element_part_id[e]
        conn = mesh.connectivity[e]
        for k in 1:N
            n = Int(conn[k])
            for (e2, _) in mesh.inverse_connectivity[n]
                p2 = layout.element_part_id[Int(e2)]
                if p2 != pe
                    push!(sets[pe], p2)
                    push!(sets[p2], pe)
                end
            end
        end
    end
    neighbors = [sort!(collect(sets[i])) for i in 1:np]
    return PartitionAdjacency(np, neighbors)
end

"""
    RankHaloExchange

Per-part halo metadata for a replicated global vector (MPI precursor).

# Fields
- `part::Int`
- `recv_neighbor`, `send_neighbor` — aligned neighbor part ids
- `recv_dof`, `send_dof` — `recv_dof[k]` global DOFs this part receives from
  `recv_neighbor[k]`; `send_dof[k]` global DOFs this part sends to
  `send_neighbor[k]`

Invariant (serial replica): `sort(send_dof[j]) == sort(recv_dof[q][…])` where
`exchanges[q]` receives from `part` at the slot matching `send_neighbor[j]`.
Checked in tests.
"""
struct RankHaloExchange
    part::Int
    recv_neighbor::Vector{Int}
    recv_dof::Vector{Vector{Int}}
    send_neighbor::Vector{Int}
    send_dof::Vector{Vector{Int}}
end

"""
    matvec_halo_mpi_request_count(exchange::RankHaloExchange) -> Int

Number of concurrent MPI operations in one [`exchange_matvec_halos_mpi!`](@ref) round:
[`length`](@ref)`(exchange.recv_neighbor)` receives plus [`length`](@ref)`(exchange.send_neighbor)`
sends. Size persistent [`MPI.Request`](@ref) buffers with
[`allocate_exchange_matvec_halo_mpi_requests`](@ref) (after `using MPI`).
"""
function matvec_halo_mpi_request_count(exchange::RankHaloExchange)::Int
    return length(exchange.recv_neighbor) + length(exchange.send_neighbor)
end

@inline function _mask_intersect!(out::BitVector, a::BitVector, b::BitVector)
    @inbounds for i in eachindex(out)
        out[i] = a[i] & b[i]
    end
    return out
end

"""
    build_rank_halo_exchanges(handler, layout, mesh, node_owner, elements)
        -> Vector{RankHaloExchange}

Setup-only routine (allocates masks and DOF lists). Uses
[`node_partition_owner_min!`](@ref)-style ownership via `node_owner`, Vertex
field layout from `handler`, and [`mark_referenced_dofs!`](@ref) /
[`mark_owned_vertex_field_dofs!`](@ref) / [`ghost_dof_mask!`](@ref).

`length(exchanges) == maximum(layout.element_part_id)`; `exchanges[p]` is the
spec for partition id `p`.

For matrix-free [`apply_K_owned_rows!`](@ref) with stencil-packed layouts, see
[`build_matvec_halo_exchanges`](@ref) in `packed_layout.jl` (included after this file).
"""
function build_rank_halo_exchanges(
    handler::DOFHandler,
    layout::MeshPartitionLayout,
    mesh::Mesh{N,T},
    node_owner::Vector{Int},
    elements::AbstractVector{El},
) where {N,T,El<:Element}
    validate_partition(layout, length(mesh.connectivity))
    ndofs = handler.total_dofs
    np = maximum(layout.element_part_id)
    length(node_owner) == length(mesh.nodes) ||
        throw(DimensionMismatch("node_owner length $(length(node_owner)), nnodes $(length(mesh.nodes))"))

    adj = build_partition_adjacency(layout, mesh)

    elem_by_part = [element_indices_for_part(layout, p) for p in 1:np]

    owned = Vector{BitVector}(undef, np)
    ghost = Vector{BitVector}(undef, np)
    ref_tmp = falses(ndofs)
    @inbounds for p in 1:np
        owned[p] = falses(ndofs)
        mark_owned_vertex_field_dofs!(owned[p], handler, node_owner, p)
        fill!(ref_tmp, false)
        mark_referenced_dofs!(ref_tmp, elements, elem_by_part[p], ndofs)
        ghost[p] = falses(ndofs)
        ghost_dof_mask!(ghost[p], ref_tmp, owned[p])
    end

    tmp = falses(ndofs)
    dof_buf = Vector{Int}(undef, ndofs)

    exchanges = Vector{RankHaloExchange}(undef, np)
    @inbounds for p in 1:np
        recv_n = Int[]
        recv_d = Vector{Int}[]
        send_n = Int[]
        send_d = Vector{Int}[]
        for q in adj.neighbors[p]
            _mask_intersect!(tmp, ghost[p], owned[q])
            nloc = collect_true_indices!(dof_buf, tmp)
            push!(recv_n, q)
            push!(recv_d, collect(Int, view(dof_buf, 1:nloc)))
        end
        for q in adj.neighbors[p]
            _mask_intersect!(tmp, ghost[q], owned[p])
            nloc = collect_true_indices!(dof_buf, tmp)
            push!(send_n, q)
            push!(send_d, collect(Int, view(dof_buf, 1:nloc)))
        end
        exchanges[p] = RankHaloExchange(p, recv_n, recv_d, send_n, send_d)
    end

    return exchanges
end

"""
    referenced_dof_mask_for_part!(
        mask::BitVector, elements, element_ids::AbstractVector{Int}, n_total_dofs::Int) -> mask

[`mark_referenced_dofs!`](@ref) alias for a fixed element-id list (reuse the
same buffer across Krylov iterations — allocation-free after setup).

    referenced_dof_mask_for_part!(
        mask::BitVector, elements, layout::MeshPartitionLayout, part::Int,
        n_total_dofs::Int) -> mask

Convenience overload that calls [`element_indices_for_part`](@ref) (allocates a
new index vector each call — fine for setup, not for inner Krylov loops).
"""
function referenced_dof_mask_for_part!(
    mask::BitVector,
    elements::AbstractVector{El},
    element_ids::AbstractVector{Int},
    n_total_dofs::Int,
) where {El <: Element}
    return mark_referenced_dofs!(mask, elements, element_ids, n_total_dofs)
end

function referenced_dof_mask_for_part!(
    mask::BitVector,
    elements::AbstractVector{El},
    layout::MeshPartitionLayout,
    part::Int,
    n_total_dofs::Int,
) where {El <: Element}
    ep = element_indices_for_part(layout, part)
    return mark_referenced_dofs!(mask, elements, ep, n_total_dofs)
end
