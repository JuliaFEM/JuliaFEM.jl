# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Partition staging buffers (`BitVector` masks + scratch vectors) must stay
allocation-free in tight loops once workspace is allocated — same contract as
the DOF-based assembler regressions.
"""

using Test
using InteractiveUtils: @allocated
using JuliaFEM
using JuliaFEM: create_elements!, @DOFSet, DOF, Displacement, Vertex
using JuliaFEM: brick_hex_partition_slabs, element_indices_for_part
using JuliaFEM: sum_element_dof_slots, mark_referenced_dofs!, collect_true_indices!
using JuliaFEM: fill_referenced_dof_indices!, ghost_dof_mask!, node_partition_owner_min!
using JuliaFEM: mark_owned_vertex_field_dofs!

@testset "partitioning workspace hot path — zero allocations" begin
    nx, ny, nz = 4, 3, 3
    mesh = create_structured_box_mesh(Hex8; nx = nx, ny = ny, nz = nz)
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    layout = brick_hex_partition_slabs(nx, ny, nz, 3; axis = :z)
    ndof = handler.total_dofs
    nnodes = length(mesh.nodes)

    ep = element_indices_for_part(layout, 2)
    @test sum_element_dof_slots(elements, ep) > 0

    mask_a = falses(ndof)
    mask_b = falses(ndof)
    mask_c = falses(ndof)
    dof_buf = Vector{Int}(undef, ndof)
    node_own = Vector{Int}(undef, nnodes)

    node_partition_owner_min!(node_own, layout, mesh)
    mark_referenced_dofs!(mask_a, elements, ep, ndof)
    collect_true_indices!(dof_buf, mask_a)
    fill_referenced_dof_indices!(dof_buf, mask_b, elements, ep, ndof)
    mark_owned_vertex_field_dofs!(mask_b, handler, node_own, 2)
    ghost_dof_mask!(mask_c, mask_a, mask_b)

    # Warm LLVM / caches
    for _ in 1:3
        fill!(mask_a, false)
        mark_referenced_dofs!(mask_a, elements, ep, ndof)
        collect_true_indices!(dof_buf, mask_a)
        fill!(mask_b, false)
        fill_referenced_dof_indices!(dof_buf, mask_b, elements, ep, ndof)
        node_partition_owner_min!(node_own, layout, mesh)
        fill!(mask_b, false)
        mark_owned_vertex_field_dofs!(mask_b, handler, node_own, 2)
        ghost_dof_mask!(mask_c, mask_a, mask_b)
    end

    @test (@allocated mark_referenced_dofs!(mask_a, elements, ep, ndof)) == 0
    @test (@allocated collect_true_indices!(dof_buf, mask_a)) == 0
    @test (@allocated fill_referenced_dof_indices!(dof_buf, mask_b, elements, ep, ndof)) == 0
    @test (@allocated node_partition_owner_min!(node_own, layout, mesh)) == 0
    @test (@allocated mark_owned_vertex_field_dofs!(mask_b, handler, node_own, 2)) == 0
    @test (@allocated ghost_dof_mask!(mask_c, mask_a, mask_b)) == 0
    @test (@allocated sum_element_dof_slots(elements, ep)) == 0
end
