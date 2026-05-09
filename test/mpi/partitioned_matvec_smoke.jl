# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md
#
# Multi-rank smoke: halo exchange + local owned-row matvec sums to global apply_K!.
# See also `partitioned_matvec_cg.jl` for unpreconditioned CG using the same MPI stack.
#
# MPI is optional (weak dependency): use an environment where both the dev package and
# MPI.jl are present, e.g. a throwaway project (same pattern as CI):
#   export MPI_SMOKE_ENV="$(mktemp -d)"
#   julia -e 'using Pkg; envdir=ENV["MPI_SMOKE_ENV"]; Pkg.activate(envdir); Pkg.develop(path=pwd()); Pkg.add("MPI"); Pkg.instantiate();
#             using MPI; run(`$(MPI.mpiexec()) -n 3 $(Base.julia_cmd()) --project=$envdir --startup-file=no $(joinpath(pwd(),"test","mpi","partitioned_matvec_smoke.jl"))`)'

using Test
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
            ContinuumFormulation{FullThreeD}(),
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
        x = zeros(n)
        if rank == 0
            Random.seed!(20260509)
            randn!(x)
        end
        MPI.Bcast!(x, 0, comm)

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
        packed = zeros(L.n_packed)
        work = zeros(L.ndofs_global)
        recv_vals = allocate_halo_recv_buffers(ex)
        send_vals = allocate_halo_send_buffers(ex)
        mpi_reqs = allocate_exchange_matvec_halo_mpi_requests(ex)

        gather_owned_from_global_to_packed!(packed, x, L)
        exchange_matvec_halos_mpi!(
            recv_vals, send_vals, packed, L, ex, comm; mpi_requests = mpi_reqs)
        unpack_halo_recv_to_packed!(packed, recv_vals, ex, L)
        fill!(work, 0.0)
        expand_packed_to_global!(work, packed, L)

        y_local = zeros(n)
        apply_K_owned_rows!(y_local, L.owned_rows, cache, asm, kernel, mesh, work)

        y_sum = zeros(n)
        MPI.Allreduce!(y_local, y_sum, MPI.SUM, comm)

        y_ref = zeros(n)
        apply_K!(y_ref, cache, asm, kernel, mesh, x)
        @test y_sum ≈ y_ref rtol = 1e-11 atol = 1e-11

        rank == 0 && println("MPI partitioned matvec smoke: OK ($(nprocs) ranks)")
    finally
        MPI.Finalize()
    end
    return nothing
end

main()
