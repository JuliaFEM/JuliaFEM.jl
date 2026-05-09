# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

using Test
using JuliaFEM
using Tensors

@testset "Mesh API (connectivity, sets, surface)" begin
    mesh = create_structured_box_mesh(Hex8, nx=2, ny=2, nz=1)
    @test validate(mesh)
    @test topology_type(mesh) === Hex8

    cm = connectivity_matrix(mesh)
    @test size(cm, 1) == nnodes(Hex8)
    @test size(cm, 2) == nelements(mesh)
    @test cm[:, 1] == collect(mesh.connectivity[1])

    p = get_node(mesh, 1)
    @test p isa Vec{3,Float64}

    ec = get_elements_for_node(mesh, UInt32(1))
    @test !isempty(ec)
    @test get_elements_for_node(mesh, 1) == ec

    els = get_elements_in_set(mesh, :all)
    @test length(els) == nelements(mesh)
    nds = get_nodes_in_set(mesh, :xmin)
    @test !isempty(nds)

    @test JuliaFEM.surface_topology(Hex8) === Quadrilateral{4}
    @test JuliaFEM.surface_topology(Tet4) === Triangle{3}

    surf = extract_surface(mesh, :all)
    @test nelements(surf) == nelements(mesh)
    @test nnodes_per_element(surf) == 4

    summary = sprint() do io
        info(io, mesh)
    end
    @test occursin("Mesh{", summary)
    @test occursin("elements", sprint(show, mesh))
end

@testset "extract_surface unsupported volume topology" begin
    nodes = [Vec((0.0, 0.0, 0.0)), Vec((1.0, 0.0, 0.0)), Vec((0.0, 1.0, 0.0)),
             Vec((0.0, 0.0, 1.0)), Vec((1.0, 0.0, 1.0)), Vec((0.0, 1.0, 1.0))]
    conn = [(UInt32(1), UInt32(2), UInt32(3), UInt32(4), UInt32(5), UInt32(6))]
    mesh_w = Mesh{Wedge{6}}(nodes, conn; element_sets=Dict(:w => Set(UInt32(1))))
    @test_throws ErrorException extract_surface(mesh_w, :w)
end
