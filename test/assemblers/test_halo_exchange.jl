# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Partition adjacency + [`RankHaloExchange`](@ref) consistency (send vs recv).
[`ReferenceMaskMultiplyLayout`](@ref) agrees with full copy when mask is all true.
"""

using Test
using InteractiveUtils: @allocated
using JuliaFEM
using JuliaFEM: ContinuumKernel, ContinuumFormulation, FullThreeD, LinearElastic
using JuliaFEM: create_elements!, @DOFSet, DOF, Displacement, Vertex
using JuliaFEM: brick_hex_partition_slabs, element_indices_for_part
using JuliaFEM: node_partition_owner_min!, build_partition_adjacency
using JuliaFEM: build_rank_halo_exchanges, build_matvec_halo_exchanges, referenced_dof_mask_for_part!
using JuliaFEM: matvec_halo_mpi_request_count
using JuliaFEM: DOFBasedCOOAssembler, DOFBasedCOOCache
using JuliaFEM: MatrixFreeOperator, LocalMultiplyLayout, ReferenceMaskMultiplyLayout
using JuliaFEM: prepare_multiply_workspace!
using LinearAlgebra: mul!

"""Every send list `p → q` matches recv list `q` receives from `p`."""
function _send_recv_consistent(exchanges::Vector{RankHaloExchange})::Bool
    np = length(exchanges)
    @inbounds for p in 1:np
        ex = exchanges[p]
        ex.part == p || return false
        for (k, q) in enumerate(ex.send_neighbor)
            sd = sort(ex.send_dof[k])
            exq = exchanges[q]
            idx = findfirst(==(p), exq.recv_neighbor)
            idx === nothing && return false
            sort(exq.recv_dof[idx]) == sd || return false
        end
    end
    return true
end

@testset "build_partition_adjacency (structured Hex8)" begin
    nx, ny, nz = 3, 2, 2
    mesh = create_structured_box_mesh(Hex8; nx = nx, ny = ny, nz = nz)
    layout = brick_hex_partition_slabs(nx, ny, nz, 2; axis = :x)
    adj = build_partition_adjacency(layout, mesh)
    @test adj.max_part_id == 2
    @test sort(union(adj.neighbors[1], adj.neighbors[2])) == [1, 2]
    @test 2 ∈ adj.neighbors[1]
    @test 1 ∈ adj.neighbors[2]
end

@testset "RankHaloExchange send/recv symmetry" begin
    nx, ny, nz = 4, 2, 2
    mesh = create_structured_box_mesh(Hex8; nx = nx, ny = ny, nz = nz)
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    layout = brick_hex_partition_slabs(nx, ny, nz, 2; axis = :x)
    nnodes = length(mesh.nodes)
    node_own = Vector{Int}(undef, nnodes)
    node_partition_owner_min!(node_own, layout, mesh)

    exch = build_rank_halo_exchanges(handler, layout, mesh, node_own, elements)
    @test length(exch) == 2
    @test _send_recv_consistent(exch)
    @test !isempty(exch[2].recv_dof[1])
end

@testset "build_matvec_halo_exchanges send/recv symmetry" begin
    nx, ny, nz = 4, 2, 2
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

    exch = build_matvec_halo_exchanges(handler, layout, mesh, node_own, elements, cache.dof_connectivity)
    @test length(exch) == 2
    @test _send_recv_consistent(exch)
end

@testset "matvec_halo_mpi_request_count" begin
    nx, ny, nz = 4, 2, 2
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
    exch = build_matvec_halo_exchanges(handler, layout, mesh, node_own, elements, cache.dof_connectivity)
    for p in 1:2
        ex = exch[p]
        @test matvec_halo_mpi_request_count(ex) ==
            length(ex.recv_neighbor) + length(ex.send_neighbor)
    end
end

@testset "matvec halo recv lists ⊇ element-patch recv lists" begin
    nx, ny, nz = 4, 2, 2
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

    ex_el = build_rank_halo_exchanges(handler, layout, mesh, node_own, elements)
    ex_mv = build_matvec_halo_exchanges(handler, layout, mesh, node_own, elements, cache.dof_connectivity)
    for p in 1:2
        el_p = ex_el[p]
        mv_p = ex_mv[p]
        @test el_p.recv_neighbor == mv_p.recv_neighbor
        for k in eachindex(el_p.recv_neighbor)
            q = el_p.recv_neighbor[k]
            idx_mv = findfirst(==(q), mv_p.recv_neighbor)
            @test idx_mv !== nothing
            @test Set(el_p.recv_dof[k]) ⊆ Set(mv_p.recv_dof[idx_mv])
        end
    end
end

@testset "ReferenceMaskMultiplyLayout — full mask ≡ LocalMultiplyLayout mul!" begin
    nx = 2
    mesh = create_structured_box_mesh(Hex8; nx = nx, ny = nx, nz = nx)
    material = LinearElastic(E = 210e9, ν = 0.3)
    kernel = ContinuumKernel(ContinuumFormulation{FullThreeD}(),
                             material, Displacement{3}())
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    asm = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elements, handler, mesh, kernel)

    nd = cache.ndofs
    mask = trues(nd)
    op_loc = MatrixFreeOperator(cache, asm, kernel, mesh; multiply_layout = LocalMultiplyLayout())
    op_ref = MatrixFreeOperator(cache, asm, kernel, mesh;
                                multiply_layout = ReferenceMaskMultiplyLayout(mask))

    x = randn(nd)
    yl = zeros(nd)
    yr = zeros(nd)
    mul!(yl, op_loc, x)
    mul!(yr, op_ref, x)
    @test yl ≈ yr
end

@testset "ReferenceMaskMultiplyLayout + referenced_dof_mask_for_part! — zero alloc" begin
    nx, ny, nz = 3, 2, 2
    mesh = create_structured_box_mesh(Hex8; nx = nx, ny = ny, nz = nz)
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    layout = brick_hex_partition_slabs(nx, ny, nz, 2; axis = :x)
    nd = handler.total_dofs
    mask = falses(nd)
    ep1 = element_indices_for_part(layout, 1)
    referenced_dof_mask_for_part!(mask, elements, ep1, nd)

    work = zeros(nd)
    x = randn(nd)
    lay = ReferenceMaskMultiplyLayout(mask)
    for _ in 1:3
        prepare_multiply_workspace!(work, x, lay)
    end
    @test (@allocated prepare_multiply_workspace!(work, x, lay)) == 0
end
