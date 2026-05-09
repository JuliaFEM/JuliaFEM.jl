# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
[`PartitionPackedLayout`](@ref): gather / expand maps, end-to-end matvec via
packed workspace + [`apply_K_owned_rows!`](@ref).
"""

using Test
using JuliaFEM
using JuliaFEM: brick_hex_partition_slabs, validate_partition
using JuliaFEM: node_partition_owner_min!
using JuliaFEM: build_partition_packed_layout, build_partition_packed_layout_for_matvec
using JuliaFEM: mark_matvec_stencil_closure!
using JuliaFEM: gather_from_global_to_packed!, expand_packed_to_global!
using JuliaFEM: gather_owned_from_global_to_packed!, gather_ghosts_from_global_to_packed!
using JuliaFEM: unpack_halo_recv_to_packed!, pack_halo_send_from_packed!
using JuliaFEM: owned_dot_packed, owned_norm²_packed, owned_dot_global_vecs
using JuliaFEM: build_rank_halo_exchanges, build_matvec_halo_exchanges
using LinearAlgebra: dot
using JuliaFEM: DOFBasedCOOAssembler, DOFBasedCOOCache
using JuliaFEM: apply_K!, apply_K_owned_rows!
using JuliaFEM: create_elements!, @DOFSet, DOF, Displacement, Vertex
using Random

@testset "build_partition_packed_layout rejects overlapping masks" begin
    v = falses(10)
    v[3] = true
    o = copy(v)
    g = copy(v)
    @test_throws ArgumentError build_partition_packed_layout(1, o, g)
end

@testset "PartitionPackedLayout gather/expand round-trip on patch" begin
    Random.seed!(20260514)
    nx, ny, nz = 3, 3, 2
    mesh = create_structured_box_mesh(Hex8; nx = nx, ny = ny, nz = nz)
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    layout = brick_hex_partition_slabs(nx, ny, nz, 2; axis = :x)
    validate_partition(layout, length(elements))
    nnodes = length(mesh.nodes)
    node_own = Vector{Int}(undef, nnodes)
    node_partition_owner_min!(node_own, layout, mesh)

    part = 2
    L = build_partition_packed_layout(handler, layout, mesh, node_own, elements, part)
    @test L.part == part
    @test L.ndofs_global == handler.total_dofs
    @test L.n_packed ≥ L.n_owned
    @test count(L.owned_rows) == L.n_owned

    x = randn(handler.total_dofs)
    packed = Vector{Float64}(undef, L.n_packed)
    gather_from_global_to_packed!(packed, x, L)
    work = zeros(handler.total_dofs)
    expand_packed_to_global!(work, packed, L)
    @inbounds for k in 1:L.n_packed
        g = L.packed_to_global[k]
        @test work[g] ≈ x[g]
    end
    # global_to_packed inverse
    @inbounds for k in 1:L.n_packed
        g = L.packed_to_global[k]
        @test L.global_to_packed[g] == k
    end
end

@testset "single-part packed work + apply_K_owned_rows! ≡ apply_K!" begin
    # `nparts == 1`: ghosts empty; element patch equals full mesh.
    Random.seed!(20260515)
    nx, ny, nz = 2, 3, 2
    mesh = create_structured_box_mesh(Hex8; nx = nx, ny = ny, nz = nz)
    material = LinearElastic(E = 210e9, ν = 0.3)
    kernel = ContinuumKernel(ContinuumFormulation{FullThreeD}(),
                             material, Displacement{3}())
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    asm = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elements, handler, mesh, kernel)
    layout = brick_hex_partition_slabs(nx, ny, nz, 1; axis = :y)
    validate_partition(layout, length(elements))
    nnodes = length(mesh.nodes)
    node_own = Vector{Int}(undef, nnodes)
    node_partition_owner_min!(node_own, layout, mesh)

    n = cache.ndofs
    x = randn(n)
    y_ref = zeros(n)
    apply_K!(y_ref, cache, asm, kernel, mesh, x)

    L = build_partition_packed_layout(handler, layout, mesh, node_own, elements, 1)
    @test L.n_packed == L.n_owned == n
    packed = Vector{Float64}(undef, L.n_packed)
    gather_from_global_to_packed!(packed, x, L)
    work = zeros(n)
    expand_packed_to_global!(work, packed, L)
    @test work ≈ x
    y = zeros(n)
    apply_K_owned_rows!(y, L.owned_rows, cache, asm, kernel, mesh, work)
    @test y ≈ y_ref rtol = 1e-11 atol = 1e-11
end

@testset "gather_owned_from_global_to_packed! only touches owned prefix" begin
    nx, ny, nz = 2, 2, 2
    mesh = create_structured_box_mesh(Hex8; nx = nx, ny = ny, nz = nz)
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    layout = brick_hex_partition_slabs(nx, ny, nz, 2; axis = :x)
    nnodes = length(mesh.nodes)
    node_own = Vector{Int}(undef, nnodes)
    node_partition_owner_min!(node_own, layout, mesh)
    L = build_partition_packed_layout(handler, layout, mesh, node_own, elements, 1)
    x = randn(handler.total_dofs)
    packed = fill!(Vector{Float64}(undef, L.n_packed), -1.0)
    gather_owned_from_global_to_packed!(packed, x, L)
    @inbounds for k in 1:L.n_owned
        @test packed[k] ≈ x[L.packed_to_global[k]]
    end
    @inbounds for k in (L.n_owned + 1):L.n_packed
        @test packed[k] == -1.0
    end
end

@testset "gather_from_global_to_packed! zero allocations after warmup" begin
    nx, ny, nz = 2, 2, 2
    mesh = create_structured_box_mesh(Hex8; nx = nx, ny = ny, nz = nz)
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    layout = brick_hex_partition_slabs(nx, ny, nz, 2; axis = :x)
    nnodes = length(mesh.nodes)
    node_own = Vector{Int}(undef, nnodes)
    node_partition_owner_min!(node_own, layout, mesh)
    L = build_partition_packed_layout(handler, layout, mesh, node_own, elements, 1)
    packed = zeros(L.n_packed)
    x = randn(handler.total_dofs)
    gather_from_global_to_packed!(packed, x, L)
    a = @allocated gather_from_global_to_packed!(packed, x, L)
    @test a == 0
    fill!(packed, 0.0)
    work = zeros(handler.total_dofs)
    expand_packed_to_global!(work, packed, L)
    b = @allocated expand_packed_to_global!(work, packed, L)
    @test b == 0
end

@testset "build_partition_packed_layout_for_matvec: multi-part sum ≡ apply_K!" begin
    Random.seed!(20260516)
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

    n = cache.ndofs
    x = randn(n)
    y_ref = zeros(n)
    apply_K!(y_ref, cache, asm, kernel, mesh, x)

    y_sum = zeros(n)
    packed = Vector{Float64}(undef, n)
    work = zeros(n)
    for p in 1:nparts
        L = build_partition_packed_layout_for_matvec(
            handler, layout, mesh, node_own, elements, p, cache.dof_connectivity)
        resize!(packed, L.n_packed)
        gather_from_global_to_packed!(packed, x, L)
        fill!(work, 0.0)
        expand_packed_to_global!(work, packed, L)
        y_p = zeros(n)
        apply_K_owned_rows!(y_p, L.owned_rows, cache, asm, kernel, mesh, work)
        y_sum .+= y_p
    end
    @test y_sum ≈ y_ref rtol = 1e-11 atol = 1e-11
end

@testset "matvec stencil closure superset of element-patch ghosts" begin
    nx, ny, nz = 4, 2, 2
    mesh = create_structured_box_mesh(Hex8; nx = nx, ny = ny, nz = nz)
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    material = LinearElastic(E = 210e9, ν = 0.3)
    kernel = ContinuumKernel(ContinuumFormulation{FullThreeD}(),
                             material, Displacement{3}())
    cache = DOFBasedCOOCache(elements, handler, mesh, kernel)
    layout = brick_hex_partition_slabs(nx, ny, nz, 2; axis = :x)
    validate_partition(layout, length(elements))
    nnodes = length(mesh.nodes)
    node_own = Vector{Int}(undef, nnodes)
    node_partition_owner_min!(node_own, layout, mesh)
    part = 2
    L_el = build_partition_packed_layout(handler, layout, mesh, node_own, elements, part)
    L_mv = build_partition_packed_layout_for_matvec(
        handler, layout, mesh, node_own, elements, part, cache.dof_connectivity)
    @test L_mv.n_packed ≥ L_el.n_packed
end

@testset "mark_matvec_stencil_closure! zero allocations after warmup" begin
    nx, ny, nz = 2, 2, 2
    mesh = create_structured_box_mesh(Hex8; nx = nx, ny = ny, nz = nz)
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    material = LinearElastic(E = 210e9, ν = 0.3)
    kernel = ContinuumKernel(ContinuumFormulation{FullThreeD}(),
                             material, Displacement{3}())
    cache = DOFBasedCOOCache(elements, handler, mesh, kernel)
    layout = brick_hex_partition_slabs(nx, ny, nz, 2; axis = :x)
    nnodes = length(mesh.nodes)
    node_own = Vector{Int}(undef, nnodes)
    node_partition_owner_min!(node_own, layout, mesh)
    nd = handler.total_dofs
    owned = falses(nd)
    mark_owned_vertex_field_dofs!(owned, handler, node_own, 1)
    closure = falses(nd)
    mark_matvec_stencil_closure!(closure, owned, elements, cache.dof_connectivity)
    a = @allocated mark_matvec_stencil_closure!(closure, owned, elements, cache.dof_connectivity)
    @test a == 0
end

@testset "gather_owned + gather_ghosts ≡ gather_from_global" begin
    nx, ny, nz = 3, 2, 2
    mesh = create_structured_box_mesh(Hex8; nx = nx, ny = ny, nz = nz)
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    layout = brick_hex_partition_slabs(nx, ny, nz, 2; axis = :x)
    nnodes = length(mesh.nodes)
    node_own = Vector{Int}(undef, nnodes)
    node_partition_owner_min!(node_own, layout, mesh)
    L = build_partition_packed_layout(handler, layout, mesh, node_own, elements, 2)
    x = randn(handler.total_dofs)
    a = zeros(L.n_packed)
    b = zeros(L.n_packed)
    gather_owned_from_global_to_packed!(a, x, L)
    gather_ghosts_from_global_to_packed!(a, x, L)
    gather_from_global_to_packed!(b, x, L)
    @test a ≈ b
end

@testset "unpack_halo_recv_to_packed! + matvec halo ≡ full gather" begin
    Random.seed!(20260517)
    nx, ny, nz = 3, 4, 2
    nparts = 3
    mesh = create_structured_box_mesh(Hex8; nx = nx, ny = ny, nz = nz)
    material = LinearElastic(E = 210e9, ν = 0.3)
    kernel = ContinuumKernel(ContinuumFormulation{FullThreeD}(),
                             material, Displacement{3}())
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    cache = DOFBasedCOOCache(elements, handler, mesh, kernel)
    layout = brick_hex_partition_slabs(nx, ny, nz, nparts; axis = :y)
    validate_partition(layout, length(elements))
    nnodes = length(mesh.nodes)
    node_own = Vector{Int}(undef, nnodes)
    node_partition_owner_min!(node_own, layout, mesh)

    exch_mv = build_matvec_halo_exchanges(
        handler, layout, mesh, node_own, elements, cache.dof_connectivity)
    x = randn(handler.total_dofs)
    for p in 1:nparts
        L = build_partition_packed_layout_for_matvec(
            handler, layout, mesh, node_own, elements, p, cache.dof_connectivity)
        ex = exch_mv[p]
        packed = zeros(L.n_packed)
        gather_owned_from_global_to_packed!(packed, x, L)
        recv_vals = [[x[g] for g in rd] for rd in ex.recv_dof]
        unpack_halo_recv_to_packed!(packed, recv_vals, ex, L)
        ref = zeros(L.n_packed)
        gather_from_global_to_packed!(ref, x, L)
        @test packed ≈ ref
    end
end

@testset "unpack_halo_recv_to_packed! / pack_halo_send_from_packed! (element patch)" begin
    nx, ny, nz = 4, 2, 2
    mesh = create_structured_box_mesh(Hex8; nx = nx, ny = ny, nz = nz)
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    layout = brick_hex_partition_slabs(nx, ny, nz, 2; axis = :x)
    nnodes = length(mesh.nodes)
    node_own = Vector{Int}(undef, nnodes)
    node_partition_owner_min!(node_own, layout, mesh)
    exch_all = build_rank_halo_exchanges(handler, layout, mesh, node_own, elements)
    x = randn(handler.total_dofs)
    for part in 1:2
        L = build_partition_packed_layout(handler, layout, mesh, node_own, elements, part)
        ex = exch_all[part]
        packed = zeros(L.n_packed)
        gather_owned_from_global_to_packed!(packed, x, L)
        recv_vals = [[x[g] for g in rd] for rd in ex.recv_dof]
        unpack_halo_recv_to_packed!(packed, recv_vals, ex, L)
        ref = zeros(L.n_packed)
        gather_from_global_to_packed!(ref, x, L)
        @test packed ≈ ref
        send_vals = [zeros(length(sd)) for sd in ex.send_dof]
        pack_halo_send_from_packed!(send_vals, ref, ex, L)
        for k in eachindex(ex.send_dof)
            sd = ex.send_dof[k]
            sv = send_vals[k]
            @test [x[g] for g in sd] ≈ sv
        end
    end
end

@testset "owned dot: partition sum ≡ global dot" begin
    nx, ny, nz = 3, 3, 2
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
    ag = randn(handler.total_dofs)
    bg = randn(handler.total_dofs)
    s_dot = 0.0
    s_norm² = 0.0
    for part in 1:2
        L = build_partition_packed_layout_for_matvec(
            handler, layout, mesh, node_own, elements, part, cache.dof_connectivity)
        s_dot += owned_dot_global_vecs(ag, bg, L)
        pa = zeros(L.n_packed)
        pb = zeros(L.n_packed)
        gather_owned_from_global_to_packed!(pa, ag, L)
        gather_owned_from_global_to_packed!(pb, bg, L)
        @test owned_dot_packed(pa, pb, L) ≈ owned_dot_global_vecs(ag, bg, L)
        s_norm² += owned_norm²_packed(pa, L)
    end
    @test s_dot ≈ dot(ag, bg)
    @test s_norm² ≈ dot(ag, ag)
end
