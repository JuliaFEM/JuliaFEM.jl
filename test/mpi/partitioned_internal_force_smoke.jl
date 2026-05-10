# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT
#
# Multi-rank smoke: `mpi_partitioned_internal_force_owned!` on disjoint owned rows,
# scattered + `MPI.Allreduce(SUM)` vs serial [`assemble_internal_force!`](@ref).
#
# Run like `partitioned_matvec_smoke.jl` (see header there); CI runs this via mpiexec.

using Test
using LinearAlgebra
using JuliaFEM
using MPI
using Random

function main()
    MPI.Init()
    try
        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)
        nprocs = MPI.Comm_size(comm)
        @test nprocs >= 2

        nparts = nprocs
        part = rank + 1

        nx, ny, nz = 3, 4, 2
        mesh = create_structured_box_mesh(Hex8; nx = nx, ny = ny, nz = nz)
        material = LinearElastic(E = 210e9, ν = 0.3)
        kernel = ContinuumKernel(
            ContinuumFormulation{ThreeDimensional}(),
            material,
            Displacement{3}(),
        )
        S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
        elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
        asm = DOFBasedCOOAssembler()
        cache = DOFBasedCOOCache(elements, handler, mesh, kernel)
        layout = brick_hex_partition_slabs(nx, ny, nz, nparts; axis = :y)
        validate_partition(layout, length(elements))
        @test maximum(layout.element_part_id) == nparts

        nnodes = length(mesh.nodes)
        node_own = Vector{Int}(undef, nnodes)
        node_partition_owner_min!(node_own, layout, mesh)

        exch_all = build_matvec_halo_exchanges(
            handler,
            layout,
            mesh,
            node_own,
            elements,
            cache.dof_connectivity,
        )

        n = cache.ndofs
        u = zeros(n)
        if rank == 0
            Random.seed!(20260522)
            randn!(u)
            u .*= 1.0e-4
        end
        MPI.Bcast!(u, 0, comm)

        L = build_partition_packed_layout_for_matvec(
            handler,
            layout,
            mesh,
            node_own,
            elements,
            part,
            cache.dof_connectivity,
        )
        ex = exch_all[part]
        ws = partitioned_mpi_owned_matvec_workspace(L, ex)
        work = zeros(n)
        mpi_reqs = allocate_exchange_matvec_halo_mpi_requests(ex)

        no = L.n_owned
        u_owned = zeros(no)
        extract_owned_subset_from_global!(u_owned, u, L)

        f_owned = zeros(no)
        mpi_partitioned_internal_force_owned!(
            f_owned,
            u_owned,
            ws.packed,
            work,
            ws.recv_vals,
            ws.send_vals,
            L,
            ex,
            cache,
            asm,
            mesh,
            comm;
            mpi_requests = mpi_reqs,
        )

        f_scat = zeros(n)
        @inbounds for k in 1:no
            g = L.packed_to_global[k]
            f_scat[g] = f_owned[k]
        end
        f_sum = similar(f_scat)
        MPI.Allreduce!(f_scat, f_sum, MPI.SUM, comm)

        f_ref = zeros(n)
        assemble_internal_force!(f_ref, cache, asm, mesh; configuration = u)
        @test f_sum ≈ f_ref rtol = 1e-10 atol = 1e-10

        rank == 0 && println(
            "MPI partitioned internal force smoke: OK ($(nprocs) ranks)",
        )
    finally
        MPI.Finalize()
    end
    return nothing
end

main()
