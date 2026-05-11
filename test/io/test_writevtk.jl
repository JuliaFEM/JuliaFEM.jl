# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

using Test
using JuliaFEM
using WriteVTK
using Tensors: Vec

@testset "write_vtu_mesh stub for non-mesh second argument" begin
    @test_throws ErrorException write_vtu_mesh("x", 1.0)
end

@testset "write_vtu_mesh Hex8 + point/cell data" begin
    mesh = create_structured_box_mesh(Hex8; xmin = 0.0, xmax = 1.0, nx = 2, ymin = 0.0, ymax = 1.0, ny = 1, zmin = 0.0, zmax = 1.0, nz = 1)
    d = mktempdir()
    base = joinpath(d, "hex8_out")
    n = nnodes_total(mesh)
    ne = nelements(mesh)
    u = randn(3, n)
    p = collect(1.0:n)
    path = write_vtu_mesh(
        base,
        mesh;
        point_data = (; displacement = u, pressure = p),
        cell_data = (; elem_id = collect(1.0:ne)),
    )
    @test endswith(path, ".vtu")
    @test isfile(path)
end

@testset "write_vtu_mesh strips .vtu suffix" begin
    mesh = create_structured_box_mesh(Hex8; xmin = 0.0, xmax = 1.0, nx = 1, ymin = 0.0, ymax = 1.0, ny = 1, zmin = 0.0, zmax = 1.0, nz = 1)
    d = mktempdir()
    path = write_vtu_mesh(joinpath(d, "with_suffix.vtu"), mesh)
    @test isfile(path)
    @test occursin("with_suffix.vtu", path)
end

@testset "write_vtu_mesh Quad4" begin
    mesh = create_structured_box_mesh(Quad4; xmin = 0.0, xmax = 1.0, nx = 2, ymin = 0.0, ymax = 1.0, ny = 2)
    d = mktempdir()
    path = write_vtu_mesh(joinpath(d, "quad"), mesh; point_data = (; s = zeros(Float64, nnodes_total(mesh))))
    @test isfile(path)
end

@testset "write_vtu_mesh unsupported topology" begin
    nodes10 = Vec{3,Float64}[
        Vec(0.0, 0.0, 0.0), Vec(1.0, 0.0, 0.0), Vec(0.0, 1.0, 0.0), Vec(0.0, 0.0, 1.0),
        Vec(0.5, 0.0, 0.0), Vec(0.5, 0.5, 0.0), Vec(0.0, 0.5, 0.0),
        Vec(0.0, 0.0, 0.5), Vec(0.5, 0.0, 0.5), Vec(0.0, 0.5, 0.5),
    ]
    conn = NTuple{10,UInt32}[
        ntuple(i -> UInt32(i), 10),
    ]
    mesh = Mesh{10,Tet10}(nodes10, conn)
    d = mktempdir()
    @test_throws ArgumentError write_vtu_mesh(joinpath(d, "t10"), mesh)
end
