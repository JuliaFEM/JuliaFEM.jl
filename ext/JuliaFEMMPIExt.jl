# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

module JuliaFEMMPIExt

using JuliaFEM
using MPI

@inline function _matvec_halo_mpi_tag(from_part::Int, to_part::Int)::Int32
    Int32((from_part % 65536) << 16 | (to_part % 65536))
end

function JuliaFEM.allocate_exchange_matvec_halo_mpi_requests(exchange::RankHaloExchange)
    n = matvec_halo_mpi_request_count(exchange)
    return Vector{MPI.Request}(undef, n)
end

function JuliaFEM.exchange_matvec_halos_mpi!(
    recv_vals::Vector{Vector{Float64}},
    send_vals::Vector{Vector{Float64}},
    packed::AbstractVector{Float64},
    layout::PartitionPackedLayout,
    exchange::RankHaloExchange,
    comm::MPI.Comm;
    mpi_requests = nothing,
)
    pack_halo_send_from_packed!(send_vals, packed, exchange, layout)
    part = exchange.part
    rank = MPI.Comm_rank(comm)
    rank + 1 == part ||
        throw(ArgumentError("MPI rank $rank inconsistent with exchange.part $part (expected rank $(part - 1))"))

    nr = length(exchange.recv_neighbor)
    ns = length(exchange.send_neighbor)
    nreq = nr + ns
    if mpi_requests === nothing
        reqs = Vector{MPI.Request}(undef, nreq)
    else
        length(mpi_requests) == nreq ||
            throw(DimensionMismatch(
                "mpi_requests length $(length(mpi_requests)), expected $nreq " *
                "(recv neighbors $nr + send neighbors $ns)",
            ))
        reqs = mpi_requests
    end

    ri = 0
    @inbounds for k in eachindex(exchange.recv_neighbor)
        q = exchange.recv_neighbor[k]
        src = q - 1
        tag = _matvec_halo_mpi_tag(q, part)
        ri += 1
        reqs[ri] = MPI.Irecv!(recv_vals[k], src, tag, comm)
    end
    @inbounds for k in eachindex(exchange.send_neighbor)
        q = exchange.send_neighbor[k]
        dest = q - 1
        tag = _matvec_halo_mpi_tag(part, q)
        ri += 1
        reqs[ri] = MPI.Isend(send_vals[k], dest, tag, comm)
    end

    MPI.Waitall(reqs)
    return nothing
end

function JuliaFEM.mpi_owned_dot_global(
    a::AbstractVector{Float64},
    b::AbstractVector{Float64},
    layout::PartitionPackedLayout,
    comm::MPI.Comm,
)::Float64
    local_s = owned_dot_global_vecs(a, b, layout)
    return MPI.Allreduce(local_s, MPI.SUM, comm)
end

function JuliaFEM.mpi_owned_dot_local(
    a_owned::AbstractVector{Float64},
    b_owned::AbstractVector{Float64},
    comm::MPI.Comm,
)::Float64
    length(a_owned) == length(b_owned) ||
        throw(DimensionMismatch("mpi_owned_dot_local: length $(length(a_owned)) != $(length(b_owned))"))
    local_s = 0.0
    @inbounds for k in eachindex(a_owned)
        local_s += a_owned[k] * b_owned[k]
    end
    return MPI.Allreduce(local_s, MPI.SUM, comm)
end

# Owned-row matvec only (no full-vector Allreduce): trial stays in `packed`, no `ndofs_global` work buffer.
function JuliaFEM.mpi_partitioned_operator_matvec_owned!(
    Ap_owned::AbstractVector{Float64},
    p_owned::AbstractVector{Float64},
    packed::AbstractVector{Float64},
    recv_vals::Vector{Vector{Float64}},
    send_vals::Vector{Vector{Float64}},
    layout::PartitionPackedLayout,
    exchange::RankHaloExchange,
    cache::DOFBasedCOOCache,
    assembler::DOFBasedCOOAssembler,
    mesh::AbstractMesh,
    comm::MPI.Comm;
    dirichlet = nothing,
    mpi_requests = nothing,
    configuration::Union{Nothing,AbstractVector{Float64}} = nothing,
    global_material_cache::Union{Nothing,GlobalMaterialCache} = nothing,
    Δt::Float64 = 0.0,
)
    copy_owned_subset_to_packed_owned_prefix!(packed, p_owned, layout)
    exchange_matvec_halos_mpi!(
        recv_vals, send_vals, packed, layout, exchange, comm; mpi_requests = mpi_requests)
    unpack_halo_recv_to_packed!(packed, recv_vals, exchange, layout)
    apply_K_owned_rows_from_packed!(
        Ap_owned, packed, layout, cache, assembler, mesh;
        configuration = configuration,
        global_material_cache = global_material_cache,
        Δt = Δt,
    )
    if dirichlet !== nothing
        dirichlet isa PenaltyDirichlet ||
            throw(ArgumentError(
                "mpi_partitioned_operator_matvec_owned! supports PenaltyDirichlet only (got $(typeof(dirichlet)))",
            ))
        apply_penalty_dirichlet_post_ap_owned!(Ap_owned, packed, layout, dirichlet)
    end
    return Ap_owned
end

function JuliaFEM.mpi_partitioned_operator_matvec_owned!(
    Ap_owned::AbstractVector{Float64},
    p_owned::AbstractVector{Float64},
    packed::AbstractVector{Float64},
    recv_vals::Vector{Vector{Float64}},
    send_vals::Vector{Vector{Float64}},
    layout::PartitionPackedLayout,
    exchange::RankHaloExchange,
    cache::DOFBasedCOOCache,
    assembler::DOFBasedCOOAssembler,
    ::AbstractKernel,
    mesh::AbstractMesh,
    comm::MPI.Comm;
    dirichlet = nothing,
    mpi_requests = nothing,
    configuration::Union{Nothing,AbstractVector{Float64}} = nothing,
    global_material_cache::Union{Nothing,GlobalMaterialCache} = nothing,
    Δt::Float64 = 0.0,
)
    JuliaFEM._depwarn_redundant_kernel_arg!(:mpi_partitioned_operator_matvec_owned!)
    return mpi_partitioned_operator_matvec_owned!(
        Ap_owned, p_owned, packed, recv_vals, send_vals, layout, exchange, cache, assembler, mesh, comm;
        dirichlet = dirichlet,
        mpi_requests = mpi_requests,
        configuration = configuration,
        global_material_cache = global_material_cache,
        Δt = Δt,
    )
end

# Owned-row internal force: pack `u`, halo exchange, then [`apply_f_int_owned_rows_from_packed!`](@ref).
# Requires `work` of length `cache.ndofs` for Pass~1 (expand + optional explicit `configuration`).
function JuliaFEM.mpi_partitioned_internal_force_owned!(
    f_int_owned::AbstractVector{Float64},
    u_owned::AbstractVector{Float64},
    packed::AbstractVector{Float64},
    work::AbstractVector{Float64},
    recv_vals::Vector{Vector{Float64}},
    send_vals::Vector{Vector{Float64}},
    layout::PartitionPackedLayout,
    exchange::RankHaloExchange,
    cache::DOFBasedCOOCache,
    assembler::DOFBasedCOOAssembler,
    mesh::AbstractMesh,
    comm::MPI.Comm;
    mpi_requests = nothing,
    configuration::Union{Nothing,AbstractVector{Float64}} = nothing,
    global_material_cache::Union{Nothing,GlobalMaterialCache} = nothing,
    Δt::Float64 = 0.0,
)
    copy_owned_subset_to_packed_owned_prefix!(packed, u_owned, layout)
    exchange_matvec_halos_mpi!(
        recv_vals, send_vals, packed, layout, exchange, comm; mpi_requests = mpi_requests)
    unpack_halo_recv_to_packed!(packed, recv_vals, exchange, layout)
    apply_f_int_owned_rows_from_packed!(
        f_int_owned, packed, work, layout, cache, assembler, mesh;
        configuration = configuration,
        global_material_cache = global_material_cache,
        Δt = Δt,
    )
    return f_int_owned
end

function JuliaFEM.mpi_partitioned_internal_force_owned!(
    f_int_owned::AbstractVector{Float64},
    u_owned::AbstractVector{Float64},
    packed::AbstractVector{Float64},
    work::AbstractVector{Float64},
    recv_vals::Vector{Vector{Float64}},
    send_vals::Vector{Vector{Float64}},
    layout::PartitionPackedLayout,
    exchange::RankHaloExchange,
    cache::DOFBasedCOOCache,
    assembler::DOFBasedCOOAssembler,
    ::AbstractKernel,
    mesh::AbstractMesh,
    comm::MPI.Comm;
    mpi_requests = nothing,
    configuration::Union{Nothing,AbstractVector{Float64}} = nothing,
    global_material_cache::Union{Nothing,GlobalMaterialCache} = nothing,
    Δt::Float64 = 0.0,
)
    JuliaFEM._depwarn_redundant_kernel_arg!(:mpi_partitioned_internal_force_owned!)
    return mpi_partitioned_internal_force_owned!(
        f_int_owned, u_owned, packed, work, recv_vals, send_vals, layout, exchange, cache, assembler, mesh, comm;
        mpi_requests = mpi_requests,
        configuration = configuration,
        global_material_cache = global_material_cache,
        Δt = Δt,
    )
end

# Global replicated matvec: owned-row stiffness + `MPI.Allreduce!`, then optional penalty BC post-hook.
function JuliaFEM.mpi_partitioned_operator_matvec!(
    Ap::AbstractVector{Float64},
    p::AbstractVector{Float64},
    packed::AbstractVector{Float64},
    work::AbstractVector{Float64},
    recv_vals::Vector{Vector{Float64}},
    send_vals::Vector{Vector{Float64}},
    layout::PartitionPackedLayout,
    exchange::RankHaloExchange,
    cache::DOFBasedCOOCache,
    assembler::DOFBasedCOOAssembler,
    mesh::AbstractMesh,
    comm::MPI.Comm;
    dirichlet = nothing,
    mpi_requests = nothing,
    configuration::Union{Nothing,AbstractVector{Float64}} = nothing,
    global_material_cache::Union{Nothing,GlobalMaterialCache} = nothing,
    Δt::Float64 = 0.0,
)
    gather_owned_from_global_to_packed!(packed, p, layout)
    exchange_matvec_halos_mpi!(
        recv_vals, send_vals, packed, layout, exchange, comm; mpi_requests = mpi_requests)
    unpack_halo_recv_to_packed!(packed, recv_vals, exchange, layout)
    fill!(work, 0.0)
    expand_packed_to_global!(work, packed, layout)
    fill!(Ap, 0.0)
    apply_K_owned_rows!(
        Ap, layout.owned_rows, cache, assembler, mesh, work;
        configuration = configuration,
        global_material_cache = global_material_cache,
        Δt = Δt,
    )
    MPI.Allreduce!(Ap, MPI.SUM, comm)
    if dirichlet !== nothing
        apply_constraint_post!(Ap, p, dirichlet)
    end
    return Ap
end

function JuliaFEM.mpi_partitioned_operator_matvec!(
    Ap::AbstractVector{Float64},
    p::AbstractVector{Float64},
    packed::AbstractVector{Float64},
    work::AbstractVector{Float64},
    recv_vals::Vector{Vector{Float64}},
    send_vals::Vector{Vector{Float64}},
    layout::PartitionPackedLayout,
    exchange::RankHaloExchange,
    cache::DOFBasedCOOCache,
    assembler::DOFBasedCOOAssembler,
    ::AbstractKernel,
    mesh::AbstractMesh,
    comm::MPI.Comm;
    dirichlet = nothing,
    mpi_requests = nothing,
    configuration::Union{Nothing,AbstractVector{Float64}} = nothing,
    global_material_cache::Union{Nothing,GlobalMaterialCache} = nothing,
    Δt::Float64 = 0.0,
)
    JuliaFEM._depwarn_redundant_kernel_arg!(:mpi_partitioned_operator_matvec!)
    return mpi_partitioned_operator_matvec!(
        Ap, p, packed, work, recv_vals, send_vals, layout, exchange, cache, assembler, mesh, comm;
        dirichlet = dirichlet,
        mpi_requests = mpi_requests,
        configuration = configuration,
        global_material_cache = global_material_cache,
        Δt = Δt,
    )
end

end # module JuliaFEMMPIExt
