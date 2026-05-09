# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

using Test
using JuliaFEM
using Tensors

@testset "Hex8 refinement" begin
    mesh = create_structured_box_mesh(Hex8, nx=1, ny=1, nz=1)
    n0 = nelements(mesh)
    refined = refine(mesh, LongestEdgeBisection(1))
    @test nelements(refined) == 2 * n0
    @test nnodes_total(refined) > nnodes_total(mesh)

    refined2 = refine(mesh, LongestEdgeBisection(2))
    @test nelements(refined2) == 4 * n0
end

@testset "compute_element_volume helper" begin
    nodes = Vec{3,Float64}[
        Vec((0.0, 0.0, 0.0)), Vec((2.0, 0.0, 0.0)), Vec((2.0, 3.0, 0.0)), Vec((0.0, 3.0, 0.0)),
        Vec((0.0, 0.0, 4.0)), Vec((2.0, 0.0, 4.0)), Vec((2.0, 3.0, 4.0)), Vec((0.0, 3.0, 4.0)),
    ]
    @test JuliaFEM.compute_element_volume(nodes) ≈ 2.0 * 3.0 * 4.0
end
