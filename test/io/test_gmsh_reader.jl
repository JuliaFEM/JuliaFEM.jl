# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

using Test
using JuliaFEM.GmshReader: read_gmsh_mesh, get_surface_nodes

@testset "GmshReader Tet4 (ASCII 4.1)" begin
    path = joinpath(@__DIR__, "..", "testdata", "cantilever_beam.msh")
    gm = read_gmsh_mesh(path)
    @test size(gm.nodes, 1) == 3
    @test size(gm.elements, 1) == 4
    @test size(gm.nodes, 2) ≥ 1
    @test size(gm.elements, 2) ≥ 1
    @test haskey(gm.physical_groups, "Bulk")

    fix = get_surface_nodes(gm, "FixedEnd")
    top = get_surface_nodes(gm, "PressureSurface")
    @test !isempty(fix)
    @test !isempty(top)
    @test isempty(get_surface_nodes(gm, "UnknownBoundary"))
end
