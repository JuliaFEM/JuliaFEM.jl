# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Pure partitioning helpers (`brick_hex_partition_slabs`, `referenced_global_dofs`,
`element_counts_by_part`) — correctness vs structured Hex8 ordering and DOF overlap
at slab interfaces.
"""

using Test
using JuliaFEM
using JuliaFEM: DOFBasedCOOAssembler, DOFBasedCOOCache
using JuliaFEM: apply_K!, apply_K_contributions!
using JuliaFEM: create_elements!, @DOFSet, DOF, Displacement, Vertex
using JuliaFEM: element_indices_for_part, validate_partition
using JuliaFEM: brick_hex_slab_upper, brick_hex_partition_slabs
using JuliaFEM: element_counts_by_part, referenced_global_dofs
using JuliaFEM: sum_element_dof_slots, mark_referenced_dofs!, collect_true_indices!
using JuliaFEM: fill_referenced_dof_indices!, ghost_dof_mask!, node_partition_owner_min!
using JuliaFEM: mark_owned_vertex_field_dofs!
using Random
using Tensors

@testset "brick_hex_partition_slabs rejects nparts > ncells on axis" begin
    @test_throws ArgumentError brick_hex_partition_slabs(1, 1, 1, 4; axis = :x)
    @test_throws ArgumentError brick_hex_partition_slabs(3, 2, 2, 4; axis = :y)
end

@testset "brick_hex_slab_upper widths" begin
    @test brick_hex_slab_upper(5, 3) == [2, 4, 5]
    @test brick_hex_slab_upper(6, 3) == [2, 4, 6]
    @test brick_hex_slab_upper(1, 4) == [1, 1, 1, 1]
end

@testset "brick_hex_partition_slabs counts + ids" begin
    nx, ny, nz = 4, 2, 2
    ne = nx * ny * nz
    layout = brick_hex_partition_slabs(nx, ny, nz, 3; axis = :x)
    validate_partition(layout, ne)
    d = element_counts_by_part(layout)
    @test sort!(collect(keys(d))) == [1, 2, 3]
    @test sum(values(d)) == ne
    # x-slab: part widths along i ≈ 2+1+1 on nx=4
    @test d[1] == 2 * ny * nz
    @test d[2] == d[3] == 1 * ny * nz
end

@testset "referenced_global_dofs overlap at x-slabs" begin
    nx, ny, nz = 4, 2, 2
    mesh = create_structured_box_mesh(Hex8; nx = nx, ny = ny, nz = nz)
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    elements, _handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    layout = brick_hex_partition_slabs(nx, ny, nz, 2; axis = :x)
    e1 = element_indices_for_part(layout, 1)
    e2 = element_indices_for_part(layout, 2)
    @assert isempty(intersect(e1, e2))
    d1 = referenced_global_dofs(elements, e1)
    d2 = referenced_global_dofs(elements, e2)
    overlap = intersect(Set(d1), Set(d2))
    @test !isempty(overlap)
end

@testset "slab-partitioned contributions sum to apply_K!" begin
    Random.seed!(20260510)
    nx, ny, nz = 3, 4, 2
    mesh = create_structured_box_mesh(Hex8; nx = nx, ny = ny, nz = nz)
    material = LinearElastic(E = 210e9, ν = 0.3)
    kernel = ContinuumKernel(ContinuumFormulation{FullThreeD}(),
                             material, Displacement{3}())
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    elements, dof_mgr = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    asm = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elements, dof_mgr, mesh, kernel)

    layout = brick_hex_partition_slabs(nx, ny, nz, 3; axis = :y)  # ny ≥ nparts
    validate_partition(layout, length(elements))

    n = cache.ndofs
    x = randn(n)
    y_full = zeros(n)
    apply_K!(y_full, cache, asm, kernel, mesh, x)

    y_sum = zeros(n)
    for part in 1:3
        ep = element_indices_for_part(layout, part)
        apply_K_contributions!(y_sum, cache, asm, kernel, mesh, x, ep)
    end
    @test y_sum ≈ y_full rtol = 1e-11 atol = 1e-11
end

@testset "fill_referenced_dof_indices! agrees with referenced_global_dofs" begin
    nx, ny, nz = 3, 2, 2
    mesh = create_structured_box_mesh(Hex8; nx = nx, ny = ny, nz = nz)
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    layout = brick_hex_partition_slabs(nx, ny, nz, 2; axis = :x)
    ep = element_indices_for_part(layout, 1)
    ndof = handler.total_dofs

    ref_alloc = referenced_global_dofs(elements, ep)

    mask = falses(ndof)
    buf = Vector{Int}(undef, ndof)
    n = fill_referenced_dof_indices!(buf, mask, elements, ep, ndof)
    @test sort(buf[1:n]) == ref_alloc
end

@testset "ghost mask nonempty for higher partition (min-node ownership)" begin
    nx, ny, nz = 4, 2, 2
    mesh = create_structured_box_mesh(Hex8; nx = nx, ny = ny, nz = nz)
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    layout = brick_hex_partition_slabs(nx, ny, nz, 2; axis = :x)
    ndof = handler.total_dofs
    nnodes = length(mesh.nodes)

    node_own = Vector{Int}(undef, nnodes)
    node_partition_owner_min!(node_own, layout, mesh)

    ep2 = element_indices_for_part(layout, 2)
    refm = falses(ndof)
    ownm = falses(ndof)
    gh = falses(ndof)

    mark_referenced_dofs!(refm, elements, ep2, ndof)
    mark_owned_vertex_field_dofs!(ownm, handler, node_own, 2)
    ghost_dof_mask!(gh, refm, ownm)
    @test any(gh)
end
