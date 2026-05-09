# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Orchestration API [`partitioned_owned_matvec!`](@ref): multi-part sum vs [`apply_K!`](@ref).
"""

using Test
using JuliaFEM
using JuliaFEM: brick_hex_partition_slabs, validate_partition
using JuliaFEM: node_partition_owner_min!
using JuliaFEM: build_partition_packed_layout_for_matvec, build_matvec_halo_exchanges
using JuliaFEM: partitioned_matvec_workspace, partitioned_mpi_owned_matvec_workspace, partitioned_owned_matvec!
using JuliaFEM: simulate_halo_recv_from_global!
using JuliaFEM: copy_owned_subset_to_packed_owned_prefix!, extract_owned_subset_from_global!, gather_owned_rows_from_workspace!
using JuliaFEM: DOFBasedCOOAssembler, DOFBasedCOOCache
using JuliaFEM: apply_K!, apply_K_owned_rows!, apply_K_owned_rows_from_packed!
using JuliaFEM: allocate_halo_recv_buffers, unpack_halo_recv_to_packed!, gather_owned_from_global_to_packed!, expand_packed_to_global!
using JuliaFEM: create_elements!, @DOFSet, DOF, Displacement, Vertex
using Random
using Tensors

@testset "partitioned_owned_matvec!: nparts sum ≡ apply_K!" begin
    Random.seed!(20260518)
    nx, ny, nz = 3, 4, 2
    nparts = 3
    mesh = create_structured_box_mesh(Hex8; nx = nx, ny = ny, nz = nz)
    material = LinearElastic(E = 210e9, ν = 0.3)
    kernel = ContinuumKernel(ContinuumFormulation{FullThreeD}(),
                             material, Displacement{3}())
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    asm = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elements, handler, mesh, kernel)
    layout = brick_hex_partition_slabs(nx, ny, nz, nparts; axis = :y)
    validate_partition(layout, length(elements))
    nnodes = length(mesh.nodes)
    node_own = Vector{Int}(undef, nnodes)
    node_partition_owner_min!(node_own, layout, mesh)

    exch_all = build_matvec_halo_exchanges(
        handler, layout, mesh, node_own, elements, cache.dof_connectivity)

    n = cache.ndofs
    x = randn(n)
    y_ref = zeros(n)
    apply_K!(y_ref, cache, asm, kernel, mesh, x)

    y_sum = zeros(n)
    for p in 1:nparts
        L = build_partition_packed_layout_for_matvec(
            handler, layout, mesh, node_own, elements, p, cache.dof_connectivity)
        ex = exch_all[p]
        ws = partitioned_matvec_workspace(L, ex)
        y_p = zeros(n)
        partitioned_owned_matvec!(
            y_p, x, ws.packed, ws.work, ws.recv_vals,
            L, ex, cache, asm, kernel, mesh)
        y_sum .+= y_p
    end
    @test y_sum ≈ y_ref rtol = 1e-11 atol = 1e-11
end

@testset "partitioned_owned_matvec! fill_recv_from_global=false" begin
    nx, ny, nz = 3, 4, 2
    nparts = 3
    mesh = create_structured_box_mesh(Hex8; nx = nx, ny = ny, nz = nz)
    material = LinearElastic(E = 210e9, ν = 0.3)
    kernel = ContinuumKernel(ContinuumFormulation{FullThreeD}(),
                             material, Displacement{3}())
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    asm = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elements, handler, mesh, kernel)
    layout = brick_hex_partition_slabs(nx, ny, nz, nparts; axis = :y)
    validate_partition(layout, length(elements))
    nnodes = length(mesh.nodes)
    node_own = Vector{Int}(undef, nnodes)
    node_partition_owner_min!(node_own, layout, mesh)
    exch_all = build_matvec_halo_exchanges(
        handler, layout, mesh, node_own, elements, cache.dof_connectivity)

    n = cache.ndofs
    x = randn(n)
    p = 2
    L = build_partition_packed_layout_for_matvec(
        handler, layout, mesh, node_own, elements, p, cache.dof_connectivity)
    ex = exch_all[p]
    ws = partitioned_matvec_workspace(L, ex)
    simulate_halo_recv_from_global!(ws.recv_vals, x, ex)
    y_a = zeros(n)
    partitioned_owned_matvec!(
        y_a, x, ws.packed, ws.work, ws.recv_vals,
        L, ex, cache, asm, kernel, mesh; fill_recv_from_global = false)
    y_b = zeros(n)
    partitioned_owned_matvec!(
        y_b, x, ws.packed, ws.work, ws.recv_vals,
        L, ex, cache, asm, kernel, mesh; fill_recv_from_global = true)
    @test y_a ≈ y_b
end

@testset "partitioned_mpi_owned_matvec_workspace rejects part mismatch" begin
    nx, ny, nz = 2, 2, 2
    mesh = create_structured_box_mesh(Hex8; nx = nx, ny = ny, nz = nz)
    material = LinearElastic(E = 210e9, ν = 0.3)
    kernel = ContinuumKernel(ContinuumFormulation{FullThreeD}(),
                             material, Displacement{3}())
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    cache = DOFBasedCOOCache(elements, handler, mesh, kernel)
    layout = brick_hex_partition_slabs(nx, ny, nz, 2; axis = :x)
    nnodes = length(mesh.nodes)
    node_own = Vector{Int}(undef, nnodes)
    node_partition_owner_min!(node_own, layout, mesh)
    exch = build_matvec_halo_exchanges(
        handler, layout, mesh, node_own, elements, cache.dof_connectivity)
    L1 = build_partition_packed_layout_for_matvec(
        handler, layout, mesh, node_own, elements, 1, cache.dof_connectivity)
    @test_throws ArgumentError partitioned_mpi_owned_matvec_workspace(L1, exch[2])
end

@testset "partitioned_matvec_workspace rejects part mismatch" begin
    nx, ny, nz = 2, 2, 2
    mesh = create_structured_box_mesh(Hex8; nx = nx, ny = ny, nz = nz)
    material = LinearElastic(E = 210e9, ν = 0.3)
    kernel = ContinuumKernel(ContinuumFormulation{FullThreeD}(),
                             material, Displacement{3}())
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    cache = DOFBasedCOOCache(elements, handler, mesh, kernel)
    layout = brick_hex_partition_slabs(nx, ny, nz, 2; axis = :x)
    nnodes = length(mesh.nodes)
    node_own = Vector{Int}(undef, nnodes)
    node_partition_owner_min!(node_own, layout, mesh)
    exch = build_matvec_halo_exchanges(
        handler, layout, mesh, node_own, elements, cache.dof_connectivity)
    L1 = build_partition_packed_layout_for_matvec(
        handler, layout, mesh, node_own, elements, 1, cache.dof_connectivity)
    @test_throws ArgumentError partitioned_matvec_workspace(L1, exch[2])
end

@testset "owned subset ↔ packed owned-prefix helpers" begin
    nx, ny, nz = 3, 4, 2
    nparts = 3
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
    nnodes = length(mesh.nodes)
    node_own = Vector{Int}(undef, nnodes)
    node_partition_owner_min!(node_own, layout, mesh)
    p = 2
    L = build_partition_packed_layout_for_matvec(
        handler, layout, mesh, node_own, elements, p, cache.dof_connectivity)
    n = cache.ndofs
    x = randn(n)
    packed_a = zeros(L.n_packed)
    packed_b = zeros(L.n_packed)
    gather_owned_from_global_to_packed!(packed_a, x, L)
    v_owned = zeros(L.n_owned)
    extract_owned_subset_from_global!(v_owned, x, L)
    copy_owned_subset_to_packed_owned_prefix!(packed_b, v_owned, L)
    @test view(packed_a, 1:L.n_owned) ≈ view(packed_b, 1:L.n_owned)

    y = zeros(n)
    apply_K!(y, cache, asm, kernel, mesh, x)
    Ap_own = zeros(L.n_owned)
    gather_owned_rows_from_workspace!(Ap_own, y, L)
    @inbounds for k in 1:L.n_owned
        g = L.packed_to_global[k]
        @test Ap_own[k] ≈ y[g]
    end
end

@testset "apply_K_owned_rows_from_packed! ≡ apply_K_owned_rows! on expanded trial" begin
    Random.seed!(20260519)
    nx, ny, nz = 3, 4, 2
    nparts = 3
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
    nnodes = length(mesh.nodes)
    node_own = Vector{Int}(undef, nnodes)
    node_partition_owner_min!(node_own, layout, mesh)
    exch_all = build_matvec_halo_exchanges(
        handler, layout, mesh, node_own, elements, cache.dof_connectivity)

    n = cache.ndofs
    x_global = randn(n)
    p = 2
    L = build_partition_packed_layout_for_matvec(
        handler, layout, mesh, node_own, elements, p, cache.dof_connectivity)
    ex = exch_all[p]
    packed = zeros(L.n_packed)
    recv_vals = allocate_halo_recv_buffers(ex)
    gather_owned_from_global_to_packed!(packed, x_global, L)
    simulate_halo_recv_from_global!(recv_vals, x_global, ex)
    unpack_halo_recv_to_packed!(packed, recv_vals, ex, L)

    work = zeros(n)
    fill!(work, 0.0)
    expand_packed_to_global!(work, packed, L)
    y_ws = zeros(n)
    apply_K_owned_rows!(y_ws, L.owned_rows, cache, asm, kernel, mesh, work)
    Ap_ref = zeros(L.n_owned)
    gather_owned_rows_from_workspace!(Ap_ref, y_ws, L)

    Ap_packed = zeros(L.n_owned)
    apply_K_owned_rows_from_packed!(Ap_packed, packed, L, cache, asm, kernel, mesh)
    @test Ap_packed ≈ Ap_ref rtol = 1e-11 atol = 1e-11
end
