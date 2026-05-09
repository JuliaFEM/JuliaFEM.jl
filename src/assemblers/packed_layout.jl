# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# Per-partition packed DOF vectors: owned block + ghost block, maps to global DOFs.

"""
    mark_matvec_stencil_closure!(
        closure::BitVector, owned::BitVector, elements, dof_connectivity::DOFConnectivity) -> closure

[`fill!`](@ref)`(closure, false)` then, for every global DOF `i` with `owned[i]`,
mark all global DOFs belonging to **any** element incident on `i` (same fan-in
as one row of [`apply_K!`](@ref)).

`length(closure) == length(owned) == dof_connectivity.n_total_dofs`. Allocation-free
given buffers (partitioning / Krylov outer setup, not first call after `fill!`).
"""
function mark_matvec_stencil_closure!(
    closure::BitVector,
    owned::BitVector,
    elements::AbstractVector{El},
    dof_connectivity::DOFConnectivity,
) where {El <: Element}
    nd = dof_connectivity.n_total_dofs
    length(closure) == nd ||
        throw(DimensionMismatch("closure length $(length(closure)), n_total_dofs $nd"))
    length(owned) == nd ||
        throw(DimensionMismatch("owned length $(length(owned)), n_total_dofs $nd"))
    isempty(elements) && throw(ArgumentError("elements must be non-empty"))

    fill!(closure, false)
    dte = dof_connectivity.dof_to_elements
    @inbounds for dof_i in 1:nd
        owned[dof_i] || continue
        conns = dte[dof_i]
        for cidx in 1:length(conns)
            conn = conns[cidx]
            ev = elem_id(conn)
            elem = elements[ev]
            for d in element_dofs(elem)
                closure[Int(d)] = true
            end
        end
    end
    return closure
end

"""
    PartitionPackedLayout

Per-partition layout for a replicated global DOF vector (`ndofs_global`).

Packed storage order is **owned DOFs first** (increasing global index), then
**ghost DOFs** (increasing global index). Ghosts are `closure \\ owned` where
`closure` is either an element-patch reference (see
[`build_partition_packed_layout`](@ref)`(handler, …)`) or a matvec row stencil
(see [`build_partition_packed_layout_for_matvec`](@ref)); both use
[`ghost_dof_mask!`](@ref).

# Fields
- `part` — partition id
- `ndofs_global` — global vector length
- `n_owned`, `n_packed` — owned count and `n_owned + n_ghost`
- `packed_to_global[k]` — global DOF index for packed slot `k`
- `global_to_packed[g]` — packed slot for global `g`, or `0` if outside this patch
- `owned_rows` — length `ndofs_global` bit mask; true on owned DOFs (for
  [`apply_K_owned_rows!`](@ref))

Use [`build_partition_packed_layout`](@ref) (element-patch ghosts, MPI halo
lists) or [`build_partition_packed_layout_for_matvec`](@ref) (row stencil
closure). Setup allocates; gather / expand are allocation-free given existing
buffers.
"""
struct PartitionPackedLayout
    part::Int
    ndofs_global::Int
    n_owned::Int
    n_packed::Int
    packed_to_global::Vector{Int}
    global_to_packed::Vector{Int}
    owned_rows::BitVector
end

function _assert_disjoint_owned_ghost(owned::BitVector, ghost::BitVector)
    length(owned) == length(ghost) ||
        throw(DimensionMismatch("owned length $(length(owned)), ghost $(length(ghost))"))
    @inbounds for i in eachindex(owned)
        (owned[i] & ghost[i]) &&
            throw(ArgumentError("owned and ghost masks overlap at global DOF index $i"))
    end
    return nothing
end

"""
    build_partition_packed_layout(part, owned::BitVector, ghost::BitVector)
        -> PartitionPackedLayout

`length(owned) == length(ghost) == ndofs_global`. Masks must be disjoint;
ghosts are typically `ghost_dof_mask!(…, referenced, owned)`.

Allocates the returned struct and copies `owned` into `layout.owned_rows`.
"""
function build_partition_packed_layout(
    part::Int,
    owned::BitVector,
    ghost::BitVector,
)::PartitionPackedLayout
    _assert_disjoint_owned_ghost(owned, ghost)
    nd = length(owned)
    buf_own = Vector{Int}(undef, nd)
    buf_gh = Vector{Int}(undef, nd)
    n_own = collect_true_indices!(buf_own, owned)
    n_gh = collect_true_indices!(buf_gh, ghost)
    n_packed = n_own + n_gh
    p2g = Vector{Int}(undef, n_packed)
    @inbounds for k in 1:n_own
        p2g[k] = buf_own[k]
    end
    @inbounds for k in 1:n_gh
        p2g[n_own + k] = buf_gh[k]
    end
    g2p = zeros(Int, nd)
    @inbounds for k in 1:n_packed
        g = p2g[k]
        g2p[g] = k
    end
    return PartitionPackedLayout(part, nd, n_own, n_packed, p2g, g2p, copy(owned))
end

"""
    build_partition_packed_layout(
        handler, layout, mesh, node_owner, elements, part::Int) -> PartitionPackedLayout

Setup helper: same owned / referenced / ghost convention as
[`build_rank_halo_exchanges`](@ref). Ghosts are DOFs on **this part's volume
elements** that are not owned here (MPI element-part halo). For a matrix-free
matvec on each part, prefer [`build_partition_packed_layout_for_matvec`](@ref).

Allocates temporary masks each call (partitioning setup, not a Krylov inner loop).
"""
function build_partition_packed_layout(
    handler::DOFHandler,
    layout::MeshPartitionLayout,
    mesh::AbstractMesh,
    node_owner::Vector{Int},
    elements::AbstractVector{El},
    part::Int,
) where {El <: Element}
    validate_partition(layout, length(elements))
    ndofs = handler.total_dofs
    ep = element_indices_for_part(layout, part)
    owned = falses(ndofs)
    mark_owned_vertex_field_dofs!(owned, handler, node_owner, part)
    ref = falses(ndofs)
    mark_referenced_dofs!(ref, elements, ep, ndofs)
    ghost = falses(ndofs)
    ghost_dof_mask!(ghost, ref, owned)
    return build_partition_packed_layout(part, owned, ghost)
end

"""
    build_partition_packed_layout_for_matvec(
        handler, layout, mesh, node_owner, elements, part, dof_connectivity)
            -> PartitionPackedLayout

Like [`build_partition_packed_layout`](@ref)`(handler, layout, …, part)`, but the
referenced set is the **row stencil closure** from
[`mark_matvec_stencil_closure!`](@ref) instead of volume elements on this part
only. Ghosts then cover every DOF needed for [`apply_K_owned_rows!`](@ref) on
this partition, so (gather → expand → matvec → sum owned rows) matches a
replicated global [`apply_K!`](@ref) up to floating-point roundoff.

Allocates temporary masks each call (setup, not an inner Krylov iteration).
"""
function build_partition_packed_layout_for_matvec(
    handler::DOFHandler,
    layout::MeshPartitionLayout,
    mesh::AbstractMesh,
    node_owner::Vector{Int},
    elements::AbstractVector{El},
    part::Int,
    dof_connectivity::DOFConnectivity,
) where {El <: Element}
    validate_partition(layout, length(elements))
    ndofs = handler.total_dofs
    dof_connectivity.n_total_dofs == ndofs ||
        throw(DimensionMismatch(
            "dof_connectivity.n_total_dofs $(dof_connectivity.n_total_dofs) != handler.total_dofs $ndofs",
        ))

    owned = falses(ndofs)
    mark_owned_vertex_field_dofs!(owned, handler, node_owner, part)
    closure = falses(ndofs)
    mark_matvec_stencil_closure!(closure, owned, elements, dof_connectivity)
    ghost = falses(ndofs)
    ghost_dof_mask!(ghost, closure, owned)
    return build_partition_packed_layout(part, owned, ghost)
end

"""
    build_matvec_halo_exchanges(handler, layout, mesh, node_owner, elements, dof_connectivity)
        -> Vector{RankHaloExchange}

Same partition adjacency and send/recv pairing as [`build_rank_halo_exchanges`](@ref),
but each part's ghost mask is `closure \\ owned` where `closure` comes from
[`mark_matvec_stencil_closure!`](@ref) (identical to ghosts in
[`build_partition_packed_layout_for_matvec`](@ref)).

Together with that packed layout, [`unpack_halo_recv_to_packed!`](@ref) fills every
ghost slot after MPI receives. Setup-only (allocates).
"""
function build_matvec_halo_exchanges(
    handler::DOFHandler,
    layout::MeshPartitionLayout,
    mesh::Mesh{N, T},
    node_owner::Vector{Int},
    elements::AbstractVector{El},
    dof_connectivity::DOFConnectivity,
) where {N, T, El <: Element}
    validate_partition(layout, length(mesh.connectivity))
    ndofs = handler.total_dofs
    dof_connectivity.n_total_dofs == ndofs ||
        throw(DimensionMismatch(
            "dof_connectivity.n_total_dofs $(dof_connectivity.n_total_dofs) != handler.total_dofs $ndofs",
        ))
    np = maximum(layout.element_part_id)
    length(node_owner) == length(mesh.nodes) ||
        throw(DimensionMismatch("node_owner length $(length(node_owner)), nnodes $(length(mesh.nodes))"))

    adj = build_partition_adjacency(layout, mesh)

    owned = Vector{BitVector}(undef, np)
    ghost = Vector{BitVector}(undef, np)
    closure_tmp = falses(ndofs)
    @inbounds for p in 1:np
        owned[p] = falses(ndofs)
        mark_owned_vertex_field_dofs!(owned[p], handler, node_owner, p)
        mark_matvec_stencil_closure!(closure_tmp, owned[p], elements, dof_connectivity)
        ghost[p] = falses(ndofs)
        ghost_dof_mask!(ghost[p], closure_tmp, owned[p])
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
    gather_from_global_to_packed!(packed, x_global, layout) -> packed

`length(packed) ≥ layout.n_packed`, `length(x_global) ≥ layout.ndofs_global`.

For each packed slot `k`, `packed[k] = x_global[packed_to_global[k]]`.

Allocation-free.
"""
function gather_from_global_to_packed!(
    packed::AbstractVector{Float64},
    x_global::AbstractVector{Float64},
    layout::PartitionPackedLayout,
)
    n = layout.n_packed
    length(packed) ≥ n ||
        throw(DimensionMismatch("packed length $(length(packed)) < n_packed $n"))
    length(x_global) ≥ layout.ndofs_global ||
        throw(DimensionMismatch("x_global length $(length(x_global)) < ndofs_global $(layout.ndofs_global)"))
    p2g = layout.packed_to_global
    @inbounds for k in 1:n
        packed[k] = x_global[p2g[k]]
    end
    return packed
end

"""
    expand_packed_to_global!(x_global, packed, layout) -> x_global

Writes `x_global[g] = packed[k]` for each mapping `g = packed_to_global[k]`.
Entries outside the patch are **not** modified.

Allocation-free.
"""
function expand_packed_to_global!(
    x_global::AbstractVector{Float64},
    packed::AbstractVector{Float64},
    layout::PartitionPackedLayout,
)
    n = layout.n_packed
    length(packed) ≥ n ||
        throw(DimensionMismatch("packed length $(length(packed)) < n_packed $n"))
    length(x_global) ≥ layout.ndofs_global ||
        throw(DimensionMismatch("x_global length $(length(x_global)) < ndofs_global $(layout.ndofs_global)"))
    p2g = layout.packed_to_global
    @inbounds for k in 1:n
        x_global[p2g[k]] = packed[k]
    end
    return x_global
end

"""
    gather_owned_from_global_to_packed!(packed, x_global, layout) -> packed

Fills only the **owned** prefix `packed[1:layout.n_owned]` from `x_global`
(increasing global DOF order, matching [`build_partition_packed_layout`](@ref)).
Ghost slots `packed[layout.n_owned+1:end]` are not touched.

Allocation-free.
"""
function gather_owned_from_global_to_packed!(
    packed::AbstractVector{Float64},
    x_global::AbstractVector{Float64},
    layout::PartitionPackedLayout,
)
    no = layout.n_owned
    length(packed) ≥ no ||
        throw(DimensionMismatch("packed length $(length(packed)) < n_owned $no"))
    length(x_global) ≥ layout.ndofs_global ||
        throw(DimensionMismatch("x_global length $(length(x_global)) < ndofs_global $(layout.ndofs_global)"))
    p2g = layout.packed_to_global
    @inbounds for k in 1:no
        packed[k] = x_global[p2g[k]]
    end
    return packed
end

"""
    copy_owned_subset_to_packed_owned_prefix!(packed, v_owned, layout) -> packed

Write `packed[k] = v_owned[k]` for `k = 1:layout.n_owned`. Ghost slots are not modified.

`length(v_owned) == layout.n_owned`. Matches lean Krylov storage keyed by the owned prefix of
[`packed_to_global`](@ref).

Allocation-free.
"""
function copy_owned_subset_to_packed_owned_prefix!(
    packed::AbstractVector{Float64},
    v_owned::AbstractVector{Float64},
    layout::PartitionPackedLayout,
)
    no = layout.n_owned
    length(v_owned) == no ||
        throw(DimensionMismatch("v_owned length $(length(v_owned)), n_owned $no"))
    length(packed) ≥ no ||
        throw(DimensionMismatch("packed length $(length(packed)) < n_owned $no"))
    @inbounds for k in 1:no
        packed[k] = v_owned[k]
    end
    return packed
end

"""
    extract_owned_subset_from_global!(v_owned, x_global, layout) -> v_owned

`v_owned[k] = x_global[packed_to_global[k]]` for `k = 1:layout.n_owned`.

Allocation-free.
"""
function extract_owned_subset_from_global!(
    v_owned::AbstractVector{Float64},
    x_global::AbstractVector{Float64},
    layout::PartitionPackedLayout,
)
    no = layout.n_owned
    length(v_owned) == no ||
        throw(DimensionMismatch("v_owned length $(length(v_owned)), n_owned $no"))
    length(x_global) ≥ layout.ndofs_global ||
        throw(DimensionMismatch(
            "x_global length $(length(x_global)) < ndofs_global $(layout.ndofs_global)",
        ))
    p2g = layout.packed_to_global
    @inbounds for k in 1:no
        v_owned[k] = x_global[p2g[k]]
    end
    return v_owned
end

"""
    gather_owned_rows_from_workspace!(v_owned, y_workspace, layout) -> v_owned

`v_owned[k] = y_workspace[packed_to_global[k]]` after e.g. [`apply_K_owned_rows!`](@ref)
(which leaves non-owned rows at zero).

Allocation-free.
"""
function gather_owned_rows_from_workspace!(
    v_owned::AbstractVector{Float64},
    y_workspace::AbstractVector{Float64},
    layout::PartitionPackedLayout,
)
    no = layout.n_owned
    nd = layout.ndofs_global
    length(v_owned) == no ||
        throw(DimensionMismatch("v_owned length $(length(v_owned)), n_owned $no"))
    length(y_workspace) ≥ nd ||
        throw(DimensionMismatch(
            "y_workspace length $(length(y_workspace)) < ndofs_global $nd",
        ))
    p2g = layout.packed_to_global
    @inbounds for k in 1:no
        v_owned[k] = y_workspace[p2g[k]]
    end
    return v_owned
end

"""
    gather_ghosts_from_global_to_packed!(packed, x_global, layout) -> packed

Fills only ghost slots `packed[layout.n_owned+1:layout.n_packed]` from
`x_global` using [`packed_to_global`](@ref). The owned prefix is not modified.

Allocation-free. Use after [`gather_owned_from_global_to_packed!`](@ref) when
simulating MPI halos from a full global replica.
"""
function gather_ghosts_from_global_to_packed!(
    packed::AbstractVector{Float64},
    x_global::AbstractVector{Float64},
    layout::PartitionPackedLayout,
)
    no = layout.n_owned
    n = layout.n_packed
    length(packed) ≥ n ||
        throw(DimensionMismatch("packed length $(length(packed)) < n_packed $n"))
    length(x_global) ≥ layout.ndofs_global ||
        throw(DimensionMismatch("x_global length $(length(x_global)) < ndofs_global $(layout.ndofs_global)"))
    p2g = layout.packed_to_global
    @inbounds for k in (no + 1):n
        packed[k] = x_global[p2g[k]]
    end
    return packed
end

"""
    unpack_halo_recv_to_packed!(packed, recv_vals, exchange, layout) -> packed

After MPI receives (or in serial tests), write neighbor halo values into
`packed` ghost slots. `recv_vals[k]` aligns with `exchange.recv_dof[k]` global
indices for neighbor `exchange.recv_neighbor[k]`.

`exchange.part` must match `layout.part`. Each global DOF must appear in
`layout.global_to_packed` with a positive index (ghost slot). Allocation-free.

Pair [`build_rank_halo_exchanges`](@ref) with [`build_partition_packed_layout`](@ref)`(handler, …)`;
pair [`build_matvec_halo_exchanges`](@ref) with [`build_partition_packed_layout_for_matvec`](@ref).
"""
function unpack_halo_recv_to_packed!(
    packed::AbstractVector{Float64},
    recv_vals::Vector{Vector{Float64}},
    exchange::RankHaloExchange,
    layout::PartitionPackedLayout,
)
    exchange.part == layout.part ||
        throw(ArgumentError("exchange.part $(exchange.part) != layout.part $(layout.part)"))
    length(recv_vals) == length(exchange.recv_dof) ==
        length(exchange.recv_neighbor) ||
        throw(DimensionMismatch("recv_vals / recv_dof / recv_neighbor length mismatch"))

    @inbounds for k in eachindex(exchange.recv_dof)
        rd = exchange.recv_dof[k]
        rv = recv_vals[k]
        length(rv) == length(rd) ||
            throw(DimensionMismatch("recv_vals[$k] length $(length(rv)) != recv_dof[$k] $(length(rd))"))
        for j in eachindex(rd)
            g = rd[j]
            (1 ≤ g ≤ layout.ndofs_global) ||
                throw(ArgumentError("recv global dof index $g out of range"))
            pk = layout.global_to_packed[g]
            pk == 0 &&
                throw(ArgumentError("recv global dof $g not in packed layout for part $(layout.part)"))
            packed[pk] = rv[j]
        end
    end
    return packed
end

"""
    pack_halo_send_from_packed!(send_vals, packed, exchange, layout) -> send_vals

Pack owned patch values into MPI send buffers. `send_vals[k][j]` receives
`packed[layout.global_to_packed[g]]` for `g = exchange.send_dof[k][j]`.

Allocation-free.
"""
function pack_halo_send_from_packed!(
    send_vals::Vector{Vector{Float64}},
    packed::AbstractVector{Float64},
    exchange::RankHaloExchange,
    layout::PartitionPackedLayout,
)
    exchange.part == layout.part ||
        throw(ArgumentError("exchange.part $(exchange.part) != layout.part $(layout.part)"))
    length(send_vals) == length(exchange.send_dof) ==
        length(exchange.send_neighbor) ||
        throw(DimensionMismatch("send_vals / send_dof / send_neighbor length mismatch"))

    @inbounds for k in eachindex(exchange.send_dof)
        sd = exchange.send_dof[k]
        sv = send_vals[k]
        length(sv) == length(sd) ||
            throw(DimensionMismatch("send_vals[$k] length $(length(sv)) != send_dof[$k] $(length(sd))"))
        for j in eachindex(sd)
            g = sd[j]
            (1 ≤ g ≤ layout.ndofs_global) ||
                throw(ArgumentError("send global dof index $g out of range"))
            pk = layout.global_to_packed[g]
            pk == 0 &&
                throw(ArgumentError("send global dof $g not in packed layout for part $(layout.part)"))
            sv[j] = packed[pk]
        end
    end
    return send_vals
end

"""
    owned_dot_packed(u, v, layout) -> Float64

Inner product over the **owned** prefix only:
`sum_{k=1}^{n_owned} u[k] * v[k]`. For Krylov on domain-decomposed vectors stored
in packed form, the global inner product is `MPI.Allreduce` of this local sum
when each global DOF is owned by exactly one rank.

Allocation-free.
"""
function owned_dot_packed(
    u::AbstractVector{Float64},
    v::AbstractVector{Float64},
    layout::PartitionPackedLayout,
)::Float64
    no = layout.n_owned
    length(u) ≥ no ||
        throw(DimensionMismatch("u length $(length(u)) < n_owned $no"))
    length(v) ≥ no ||
        throw(DimensionMismatch("v length $(length(v)) < n_owned $no"))
    s = 0.0
    @inbounds for k in 1:no
        s += u[k] * v[k]
    end
    return s
end

"""
    owned_norm²_packed(u, layout) -> Float64

[`owned_dot_packed`](@ref)`(u, u, layout)`.
"""
function owned_norm²_packed(u::AbstractVector{Float64}, layout::PartitionPackedLayout)::Float64
    return owned_dot_packed(u, u, layout)
end

"""
    owned_dot_global_vecs(a_global, b_global, layout) -> Float64

`sum_{g : layout.owned_rows[g]} a_global[g] * b_global[g]`. Matches
[`owned_dot_packed`](@ref) when `a_global` / `b_global` are full-length globals
and `u`/`v` hold the owned entries in packed global-index order.

Allocation-free.
"""
function owned_dot_global_vecs(
    a_global::AbstractVector{Float64},
    b_global::AbstractVector{Float64},
    layout::PartitionPackedLayout,
)::Float64
    length(a_global) ≥ layout.ndofs_global ||
        throw(DimensionMismatch("a_global length $(length(a_global)) < ndofs_global $(layout.ndofs_global)"))
    length(b_global) ≥ layout.ndofs_global ||
        throw(DimensionMismatch("b_global length $(length(b_global)) < ndofs_global $(layout.ndofs_global)"))
    ow = layout.owned_rows
    length(ow) == layout.ndofs_global ||
        throw(DimensionMismatch("owned_rows length mismatch"))
    s = 0.0
    @inbounds for g in 1:layout.ndofs_global
        ow[g] || continue
        s += a_global[g] * b_global[g]
    end
    return s
end
