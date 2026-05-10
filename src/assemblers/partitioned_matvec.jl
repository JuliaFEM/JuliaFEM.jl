# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

# Single-partition orchestration for matrix-free `K*x` and owned-row internal
# force using a packed patch + matvec halo exchange (MPI-ready).

"""
    allocate_halo_recv_buffers(exchange::RankHaloExchange) -> Vector{Vector{Float64}}

Preallocate receive buffers aligned with `exchange.recv_dof` (setup only).

After MPI non-blocking receives complete, or [`simulate_halo_recv_from_global!`](@ref)
in serial, pass the result
to [`unpack_halo_recv_to_packed!`](@ref).
"""
function allocate_halo_recv_buffers(exchange::RankHaloExchange)::Vector{Vector{Float64}}
    return [zeros(length(rd)) for rd in exchange.recv_dof]
end

"""
    allocate_halo_send_buffers(exchange::RankHaloExchange) -> Vector{Vector{Float64}}

Preallocate send buffers aligned with `exchange.send_dof` (setup only).

Used by [`pack_halo_send_from_packed!`](@ref) before [`exchange_matvec_halos_mpi!`](@ref)
(with both `JuliaFEM` and `MPI` loaded).
"""
function allocate_halo_send_buffers(exchange::RankHaloExchange)::Vector{Vector{Float64}}
    return [zeros(length(sd)) for sd in exchange.send_dof]
end

"""
    partitioned_matvec_workspace(layout::PartitionPackedLayout, exchange::RankHaloExchange)

Named tuple `(packed, work, recv_vals)` with correct lengths for
[`partitioned_owned_matvec!`](@ref):

- `packed` — length `layout.n_packed`
- `work` — length `layout.ndofs_global` (expanded stencil workspace)
- `recv_vals` — from [`allocate_halo_recv_buffers`](@ref)`(exchange)`

`exchange.part` must match `layout.part`.
"""
function partitioned_matvec_workspace(
    layout::PartitionPackedLayout,
    exchange::RankHaloExchange,
)
    exchange.part == layout.part ||
        throw(ArgumentError("exchange.part $(exchange.part) != layout.part $(layout.part)"))
    packed = zeros(layout.n_packed)
    work = zeros(layout.ndofs_global)
    recv_vals = allocate_halo_recv_buffers(exchange)
    return (; packed, work, recv_vals)
end

"""
    partitioned_mpi_owned_matvec_workspace(layout::PartitionPackedLayout, exchange::RankHaloExchange)

Like [`partitioned_matvec_workspace`](@ref), but omits the `ndofs_global`-length `work` buffer used
by [`expand_packed_to_global!`](@ref). Intended for the lean MPI path
[`mpi_partitioned_operator_matvec_owned!`](@ref), which applies the stiffness rows directly from
[`packed`](@ref).

Returns `(; packed, recv_vals, send_vals)` with `exchange.part == layout.part`.
"""
function partitioned_mpi_owned_matvec_workspace(
    layout::PartitionPackedLayout,
    exchange::RankHaloExchange,
)
    exchange.part == layout.part ||
        throw(ArgumentError("exchange.part $(exchange.part) != layout.part $(layout.part)"))
    packed = zeros(layout.n_packed)
    recv_vals = allocate_halo_recv_buffers(exchange)
    send_vals = allocate_halo_send_buffers(exchange)
    return (; packed, recv_vals, send_vals)
end

"""
    simulate_halo_recv_from_global!(recv_vals, x_global, exchange) -> recv_vals

Serial stand-in for completed MPI receives:

`recv_vals[k][j] = x_global[exchange.recv_dof[k][j]]`.

Buffers must be preallocated (`recv_vals[k]` length matches `recv_dof[k]`).
Allocation-free in the inner loops.
"""
function simulate_halo_recv_from_global!(
    recv_vals::Vector{Vector{Float64}},
    x_global::AbstractVector{Float64},
    exchange::RankHaloExchange,
)
    length(recv_vals) == length(exchange.recv_dof) ==
        length(exchange.recv_neighbor) ||
        throw(DimensionMismatch("recv_vals length mismatch"))
    @inbounds for k in eachindex(exchange.recv_dof)
        rd = exchange.recv_dof[k]
        rv = recv_vals[k]
        length(rv) == length(rd) ||
            throw(DimensionMismatch("recv_vals[$k] length $(length(rv)) != recv_dof length $(length(rd))"))
        for j in eachindex(rd)
            g = rd[j]
            rv[j] = x_global[g]
        end
    end
    return recv_vals
end

"""
    partitioned_owned_matvec!(
        y_contrib, x_global, packed, work, recv_vals,
        layout, exchange, cache, assembler, mesh;
        fill_recv_from_global = true,
        configuration = nothing, global_material_cache = nothing, Δt = 0.0,
    ) -> y_contrib
    partitioned_owned_matvec!(
        y_contrib, x_global, packed, work, recv_vals,
        layout, exchange, cache, assembler, kernel, mesh;
        fill_recv_from_global = true,
        configuration = nothing, global_material_cache = nothing, Δt = 0.0,
    ) -> y_contrib

One partition’s additive contribution to `K*x` on **owned rows** (zeros elsewhere
in `y_contrib`), using the matvec packed layout and [`build_matvec_halo_exchanges`](@ref)
metadata:

1. [`gather_owned_from_global_to_packed!`](@ref)`(packed, x_global, layout)`
2. If `fill_recv_from_global` — [`simulate_halo_recv_from_global!`](@ref)`(recv_vals, x_global, exchange)` (skip when MPI fills `recv_vals`)
3. [`unpack_halo_recv_to_packed!`](@ref)`(packed, recv_vals, exchange, layout)`
4. [`fill!`](@ref)`(work, 0)` then [`expand_packed_to_global!`](@ref)`(work, packed, layout)`
5. [`apply_K_owned_rows!`](@ref)`(y_contrib, layout.owned_rows, …; configuration, …)`

Optional keywords `configuration`, `global_material_cache`, and `Δt` match
[`apply_K!`](@ref) / [`apply_K_owned_rows!`](@ref) (nonlinear continuum Pass~1).

Requires `exchange.part == layout.part`, `length(y_contrib) == cache.ndofs`, and
matching `work` / `packed` sizes from [`partitioned_matvec_workspace`](@ref).

Summing `y_contrib` over disjoint owned partitions recovers global [`apply_K!`](@ref)
when `x_global` is the full vector (serial replica).

Allocation-free after workspace setup when `fill_recv_from_global` follows the same pattern.

The form with `kernel` ignores it (backward compatibility).
"""
function partitioned_owned_matvec!(
    y_contrib::AbstractVector{Float64},
    x_global::AbstractVector{Float64},
    packed::AbstractVector{Float64},
    work::AbstractVector{Float64},
    recv_vals::Vector{Vector{Float64}},
    layout::PartitionPackedLayout,
    exchange::RankHaloExchange,
    cache::DOFBasedCOOCache,
    assembler::DOFBasedCOOAssembler,
    mesh::AbstractMesh;
    fill_recv_from_global::Bool = true,
    configuration::Union{Nothing,AbstractVector{Float64}} = nothing,
    global_material_cache::Union{Nothing,GlobalMaterialCache} = nothing,
    Δt::Float64 = 0.0,
)
    exchange.part == layout.part ||
        throw(ArgumentError("exchange.part $(exchange.part) != layout.part $(layout.part)"))
    nd = cache.ndofs
    length(y_contrib) == nd ||
        throw(DimensionMismatch("y_contrib length $(length(y_contrib)), ndofs $nd"))
    length(x_global) == nd ||
        throw(DimensionMismatch("x_global length $(length(x_global)), ndofs $nd"))
    length(work) == nd ||
        throw(DimensionMismatch("work length $(length(work)), ndofs $nd"))
    layout.ndofs_global == nd ||
        throw(DimensionMismatch("layout.ndofs_global $(layout.ndofs_global), cache.ndofs $nd"))

    gather_owned_from_global_to_packed!(packed, x_global, layout)
    if fill_recv_from_global
        simulate_halo_recv_from_global!(recv_vals, x_global, exchange)
    end
    unpack_halo_recv_to_packed!(packed, recv_vals, exchange, layout)
    fill!(work, 0.0)
    expand_packed_to_global!(work, packed, layout)
    apply_K_owned_rows!(
        y_contrib, layout.owned_rows, cache, assembler, mesh, work;
        configuration = configuration,
        global_material_cache = global_material_cache,
        Δt = Δt,
    )
    return y_contrib
end

function partitioned_owned_matvec!(
    y_contrib::AbstractVector{Float64},
    x_global::AbstractVector{Float64},
    packed::AbstractVector{Float64},
    work::AbstractVector{Float64},
    recv_vals::Vector{Vector{Float64}},
    layout::PartitionPackedLayout,
    exchange::RankHaloExchange,
    cache::DOFBasedCOOCache,
    assembler::DOFBasedCOOAssembler,
    ::AbstractKernel,
    mesh::AbstractMesh;
    fill_recv_from_global::Bool = true,
    configuration::Union{Nothing,AbstractVector{Float64}} = nothing,
    global_material_cache::Union{Nothing,GlobalMaterialCache} = nothing,
    Δt::Float64 = 0.0,
)
    _depwarn_redundant_kernel_arg!(:partitioned_owned_matvec!)
    return partitioned_owned_matvec!(
        y_contrib, x_global, packed, work, recv_vals, layout, exchange, cache, assembler, mesh;
        fill_recv_from_global = fill_recv_from_global,
        configuration = configuration,
        global_material_cache = global_material_cache,
        Δt = Δt,
    )
end

"""
    partitioned_owned_internal_force!(
        y_contrib, u_global, packed, work, recv_vals,
        layout, exchange, cache, assembler, mesh;
        fill_recv_from_global = true,
        configuration = nothing, global_material_cache = nothing, Δt = 0.0,
    ) -> y_contrib
    partitioned_owned_internal_force!(
        y_contrib, u_global, packed, work, recv_vals,
        layout, exchange, cache, assembler, kernel, mesh;
        fill_recv_from_global = true,
        configuration = nothing, global_material_cache = nothing, Δt = 0.0,
    ) -> y_contrib

One partition’s additive contribution to the nonlinear internal force
[`assemble_internal_force!`](@ref) on **owned rows** (zeros elsewhere in
`y_contrib`), using the same packed patch and halo metadata as
[`partitioned_owned_matvec!`](@ref):

1. [`gather_owned_from_global_to_packed!`](@ref)`(packed, u_global, layout)`
2. If `fill_recv_from_global` — [`simulate_halo_recv_from_global!`](@ref)`(recv_vals, u_global, exchange)`
3. [`unpack_halo_recv_to_packed!`](@ref)
4. [`fill!`](@ref)`(work, 0)` then [`expand_packed_to_global!`](@ref)`(work, packed, layout)`
5. [`apply_f_int_owned_rows!`](@ref)`(y_contrib, layout.owned_rows, …; configuration, …)`

When `configuration === nothing`, Pass~1 uses the expanded patch displacement
`work`. Otherwise `configuration` is forwarded to [`_prepare_caches!`](@ref)
explicitly (length `cache.ndofs`).

The form with `kernel` ignores it (backward compatibility).
"""
function partitioned_owned_internal_force!(
    y_contrib::AbstractVector{Float64},
    u_global::AbstractVector{Float64},
    packed::AbstractVector{Float64},
    work::AbstractVector{Float64},
    recv_vals::Vector{Vector{Float64}},
    layout::PartitionPackedLayout,
    exchange::RankHaloExchange,
    cache::DOFBasedCOOCache,
    assembler::DOFBasedCOOAssembler,
    mesh::AbstractMesh;
    fill_recv_from_global::Bool = true,
    configuration::Union{Nothing,AbstractVector{Float64}} = nothing,
    global_material_cache::Union{Nothing,GlobalMaterialCache} = nothing,
    Δt::Float64 = 0.0,
)
    exchange.part == layout.part ||
        throw(ArgumentError("exchange.part $(exchange.part) != layout.part $(layout.part)"))
    nd = cache.ndofs
    length(y_contrib) == nd ||
        throw(DimensionMismatch("y_contrib length $(length(y_contrib)), ndofs $nd"))
    length(u_global) == nd ||
        throw(DimensionMismatch("u_global length $(length(u_global)), ndofs $nd"))
    length(work) == nd ||
        throw(DimensionMismatch("work length $(length(work)), ndofs $nd"))
    layout.ndofs_global == nd ||
        throw(DimensionMismatch("layout.ndofs_global $(layout.ndofs_global), cache.ndofs $nd"))

    gather_owned_from_global_to_packed!(packed, u_global, layout)
    if fill_recv_from_global
        simulate_halo_recv_from_global!(recv_vals, u_global, exchange)
    end
    unpack_halo_recv_to_packed!(packed, recv_vals, exchange, layout)
    fill!(work, 0.0)
    expand_packed_to_global!(work, packed, layout)
    prep_cfg = configuration === nothing ? work : configuration
    apply_f_int_owned_rows!(
        y_contrib, layout.owned_rows, cache, assembler, mesh;
        configuration = prep_cfg,
        global_material_cache = global_material_cache,
        Δt = Δt,
    )
    return y_contrib
end

function partitioned_owned_internal_force!(
    y_contrib::AbstractVector{Float64},
    u_global::AbstractVector{Float64},
    packed::AbstractVector{Float64},
    work::AbstractVector{Float64},
    recv_vals::Vector{Vector{Float64}},
    layout::PartitionPackedLayout,
    exchange::RankHaloExchange,
    cache::DOFBasedCOOCache,
    assembler::DOFBasedCOOAssembler,
    ::AbstractKernel,
    mesh::AbstractMesh;
    fill_recv_from_global::Bool = true,
    configuration::Union{Nothing,AbstractVector{Float64}} = nothing,
    global_material_cache::Union{Nothing,GlobalMaterialCache} = nothing,
    Δt::Float64 = 0.0,
)
    _depwarn_redundant_kernel_arg!(:partitioned_owned_internal_force!)
    return partitioned_owned_internal_force!(
        y_contrib, u_global, packed, work, recv_vals, layout, exchange, cache, assembler, mesh;
        fill_recv_from_global = fill_recv_from_global,
        configuration = configuration,
        global_material_cache = global_material_cache,
        Δt = Δt,
    )
end

"""
    apply_K_owned_rows_from_packed!(
        Ap_owned, packed, layout, cache, assembler, mesh,
    ) -> Ap_owned
    apply_K_owned_rows_from_packed!(
        Ap_owned, packed, layout, cache, assembler, kernel, mesh,
    ) -> Ap_owned

Like [`apply_K_owned_rows!`](@ref)`(y, layout.owned_rows, …, x)` with trial vector `x` implied by
`packed`: for each global column index `j`, use `packed[layout.global_to_packed[j]]`. Only owned
rows `packed_to_global[1:n_owned]` are computed; results go to `Ap_owned[k]` for row
`packed_to_global[k]`.

Requires every stencil neighbor `j` of those rows to satisfy `layout.global_to_packed[j] ≠ 0`
(as ensured by [`build_partition_packed_layout_for_matvec`](@ref)).

`length(Ap_owned) == layout.n_owned`, `length(packed) ≥ layout.n_packed`, `layout.ndofs_global == cache.ndofs`.
Allocation-free after warmup.

The form with `kernel` ignores it (backward compatibility).
"""
function apply_K_owned_rows_from_packed!(
    Ap_owned::AbstractVector{Float64},
    packed::AbstractVector{Float64},
    layout::PartitionPackedLayout,
    cache::DOFBasedCOOCache{T,B,IPS,E,GC,Buf,FieldType,StateType,KS},
    assembler::DOFBasedCOOAssembler,
    mesh::AbstractMesh;
    configuration::Union{Nothing,AbstractVector{Float64}} = nothing,
    global_material_cache::Union{Nothing,GlobalMaterialCache} = nothing,
    Δt::Float64 = 0.0,
) where {T,B,IPS,E<:AbstractElement,GC,Buf,FieldType,StateType,KS}
    ndofs = cache.ndofs
    layout.ndofs_global == ndofs ||
        throw(DimensionMismatch(
            "layout.ndofs_global $(layout.ndofs_global) != cache.ndofs $ndofs",
        ))
    no = layout.n_owned
    n_packed = layout.n_packed
    length(Ap_owned) == no ||
        throw(DimensionMismatch("Ap_owned length $(length(Ap_owned)), n_owned $no"))
    length(packed) ≥ n_packed ||
        throw(DimensionMismatch("packed length $(length(packed)) < n_packed $n_packed"))

    owned_rows = layout.owned_rows
    length(owned_rows) == ndofs ||
        throw(DimensionMismatch("owned_rows length $(length(owned_rows)); expected $ndofs"))

    g2p = layout.global_to_packed
    p2g = layout.packed_to_global

    _prepare_caches!(
        cache, mesh;
        configuration = configuration,
        global_material_cache = global_material_cache,
        Δt = Δt,
    )

    elements         = cache.elements
    element_caches   = cache.element_caches
    geometry_caches  = cache.geometry_caches
    qp_buffers       = cache.qp_buffers

    dof_connectivity = cache.dof_connectivity
    dof_to_elements  = dof_connectivity.dof_to_elements

    loc_layout = local_dof_layout(E)
    ndofs_elem = length(loc_layout)

    @inbounds for k in 1:no
        dof_i = p2g[k]
        owned_rows[dof_i] ||
            throw(ArgumentError("packed owned slot $k → global $dof_i not marked owned in layout"))

        yi                = 0.0
        touching_elements = dof_to_elements[dof_i]
        n_conns           = length(touching_elements)

        @inbounds for conn_idx in 1:n_conns
            conn        = touching_elements[conn_idx]
            elem_id_val = elem_id(conn)
            local_i     = local_dof_idx(conn)

            element        = elements[elem_id_val]::E
            element_cache  = element_caches[elem_id_val]
            geometry_cache = geometry_caches[elem_id_val]
            qp_buffer      = view(qp_buffers, :, elem_id_val)

            entry_i      = loc_layout[local_i]
            dofs_elem    = element_cache.dofs
            k_e          = kernel_at(cache, Int(elem_id_val))

            @inbounds for local_j in 1:ndofs_elem
                dof_j_global = Int(dofs_elem[local_j])
                entry_j      = loc_layout[local_j]

                K_ij = evaluate_entry(
                    k_e,
                    geometry_cache,
                    qp_buffer,
                    entry_i,
                    entry_j,
                    Int(elem_id_val),
                )

                pk_j = g2p[dof_j_global]
                pk_j == 0 &&
                    throw(ArgumentError(
                        "global trial dof $dof_j_global not in packed patch (row $dof_i)",
                    ))
                yi += K_ij * packed[pk_j]
            end
        end

        Ap_owned[k] = yi
    end

    return Ap_owned
end

function apply_K_owned_rows_from_packed!(
    Ap_owned::AbstractVector{Float64},
    packed::AbstractVector{Float64},
    layout::PartitionPackedLayout,
    cache::DOFBasedCOOCache{T,B,IPS,E,GC,Buf,FieldType,StateType,KS},
    assembler::DOFBasedCOOAssembler,
    ::AbstractKernel,
    mesh::AbstractMesh;
    configuration::Union{Nothing,AbstractVector{Float64}} = nothing,
    global_material_cache::Union{Nothing,GlobalMaterialCache} = nothing,
    Δt::Float64 = 0.0,
) where {T,B,IPS,E<:AbstractElement,GC,Buf,FieldType,StateType,KS}
    _depwarn_redundant_kernel_arg!(:apply_K_owned_rows_from_packed!)
    return apply_K_owned_rows_from_packed!(
        Ap_owned, packed, layout, cache, assembler, mesh;
        configuration = configuration,
        global_material_cache = global_material_cache,
        Δt = Δt,
    )
end
