# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md
#
# Multi-rank unpreconditioned CG on heat conduction (same BC/source as the serial matrix-free
# regression): lean owned-only Krylov vectors + halo matvec **without** a full-vector matvec
# `Allreduce`. Inner products use `mpi_owned_dot_local` (scalar `Allreduce` only).
#
# Run with the same throwaway-env pattern as `partitioned_matvec_smoke.jl` (see file header there).

using Test
using LinearAlgebra
using JuliaFEM
using MPI

function main()
    MPI.Init()
    try
        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)
        nprocs = MPI.Comm_size(comm)
        @test nprocs >= 2

        nparts = nprocs
        part = rank + 1

        nx = ny = nz = 3
        mesh = create_structured_box_mesh(Hex8; nx = nx, ny = ny, nz = nz)
        material = HeatConductivity(k = 50.2)
        kernel = HeatKernel(ContinuumFormulation{FullThreeD}(), material)
        S = @DOFSet{T::DOF{Temperature, Vertex}}
        elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
        asm = DOFBasedCOOAssembler()
        cache = DOFBasedCOOCache(elements, handler, mesh, kernel)

        fixed_dofs = Int[]
        for (nid, X) in enumerate(mesh.nodes)
            if X[3] == 0.0
                push!(fixed_dofs, nid)
            end
        end
        @test !isempty(fixed_dofs)
        bc = PenaltyDirichlet(fixed_dofs; penalty = 1e16)

        n = cache.ndofs
        top_corner_node = (nx + 1) * (ny + 1) * (nz + 1)
        b_full = zeros(n)
        b_full[top_corner_node] = 1.0
        apply_constraint!(b_full, bc)

        layout_part = brick_hex_partition_slabs(nx, ny, nz, nparts; axis = :y)
        validate_partition(layout_part, length(elements))
        @test maximum(layout_part.element_part_id) == nparts

        nnodes = length(mesh.nodes)
        node_own = Vector{Int}(undef, nnodes)
        node_partition_owner_min!(node_own, layout_part, mesh)

        exch_all = build_matvec_halo_exchanges(
            handler,
            layout_part,
            mesh,
            node_own,
            elements,
            cache.dof_connectivity,
        )

        L = build_partition_packed_layout_for_matvec(
            handler,
            layout_part,
            mesh,
            node_own,
            elements,
            part,
            cache.dof_connectivity,
        )
        ex = exch_all[part]
        no = L.n_owned
        ws = partitioned_mpi_owned_matvec_workspace(L, ex)
        mpi_reqs = allocate_exchange_matvec_halo_mpi_requests(ex)

        b_owned = zeros(no)
        extract_owned_subset_from_global!(b_owned, b_full, L)

        u_ref = zeros(n)
        if rank == 0
            assemble!(cache, asm, kernel, mesh)
            K, _ = extract_system(cache)
            Kbc = Matrix(K)
            apply_constraint!(Kbc, bc)
            u_ref .= Kbc \ b_full
        end
        MPI.Bcast!(u_ref, 0, comm)

        u_ref_owned = zeros(no)
        extract_owned_subset_from_global!(u_ref_owned, u_ref, L)

        x_owned = zeros(no)
        r_owned = copy(b_owned)
        p_owned = copy(r_owned)
        Ap_owned = zeros(no)

        rsold = mpi_owned_dot_local(r_owned, r_owned, comm)
        tol² = (1e-10)^2
        maxiter = 4 * n
        iter_out = 0
        for k in 1:maxiter
            iter_out = k
            mpi_partitioned_operator_matvec_owned!(
                Ap_owned, p_owned, ws.packed,
                ws.recv_vals, ws.send_vals, L, ex, cache, asm, kernel, mesh, comm;
                dirichlet = bc,
                mpi_requests = mpi_reqs,
            )
            denom = mpi_owned_dot_local(p_owned, Ap_owned, comm)
            α = rsold / denom
            x_owned .+= α .* p_owned
            r_owned .-= α .* Ap_owned
            rsnew = mpi_owned_dot_local(r_owned, r_owned, comm)
            if rsnew < tol²
                break
            end
            β = rsnew / rsold
            p_owned .= r_owned .+ β .* p_owned
            rsold = rsnew
        end

        @test iter_out < maxiter

        diff_owned = similar(x_owned)
        diff_owned .= x_owned .- u_ref_owned
        err² = mpi_owned_dot_local(diff_owned, diff_owned, comm)
        scale² = mpi_owned_dot_local(u_ref_owned, u_ref_owned, comm)
        rel_err = sqrt(err²) / max(sqrt(scale²), 1.0)
        @test rel_err < 1e-5

        mpi_partitioned_operator_matvec_owned!(
            Ap_owned, x_owned, ws.packed,
            ws.recv_vals, ws.send_vals, L, ex, cache, asm, kernel, mesh, comm;
            dirichlet = bc,
            mpi_requests = mpi_reqs,
        )
        res_owned = similar(b_owned)
        res_owned .= b_owned .- Ap_owned
        res² = mpi_owned_dot_local(res_owned, res_owned, comm)
        b² = mpi_owned_dot_local(b_owned, b_owned, comm)
        rel_res = sqrt(res²) / max(sqrt(b²), 1.0)
        @test rel_res < 1e-5

        rank == 0 && println(
            "MPI lean partitioned CG (heat): OK ($(nprocs) ranks, $(iter_out) iters, rel_err=$(round(rel_err; sigdigits=3)), rel_res=$(round(rel_res; sigdigits=3)))",
        )
    finally
        MPI.Finalize()
    end
    return nothing
end

main()
