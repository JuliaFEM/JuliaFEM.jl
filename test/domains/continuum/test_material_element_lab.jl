# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

using Test
using LinearAlgebra
using JuliaFEM

@testset "Material element lab (symmetric uniaxial Hex8 coupon)" begin
    L = 1.0
    mesh = material_lab_single_hex8_brick(; L = L)
    @test nelements(mesh) == 1

    S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})

    δx = 1.0e-4
    E = 210e9
    ν = 0.3
    u = material_lab_linear_elastic_uniaxial_solve(mesh, handler, elements, E, ν, δx)

    @test all(isfinite, u)
    for nid in get_nodes_in_set(mesh, :xmax)
        ud = get_node_dofs(handler, Int(nid))
        @test u[ud[1]] ≈ δx rtol = 1e-8 atol = 1e-12
    end
    for nid in get_nodes_in_set(mesh, :xmin)
        ud = get_node_dofs(handler, Int(nid))
        @test abs(u[ud[1]]) < 1e-12
    end
    for nid in get_nodes_in_set(mesh, :ymin)
        ud = get_node_dofs(handler, Int(nid))
        @test abs(u[ud[2]]) < 1e-12
    end
    for nid in get_nodes_in_set(mesh, :zmin)
        ud = get_node_dofs(handler, Int(nid))
        @test abs(u[ud[3]]) < 1e-12
    end

    bc = hex8_symmetric_uniaxial_eliminated_dirichlet(mesh, handler, δx)
    @test bc isa EliminatedDirichlet
    @test length(bc.fixed_dofs) == length(bc.values)
end
