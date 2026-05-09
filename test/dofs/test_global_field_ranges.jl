# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using Test
using JuliaFEM
using JuliaFEM: create_unit_cube_mesh, create_elements!, @DOFSet, DOF, Displacement, Vertex
using JuliaFEM: global_field_ranges, saddle_point_blocks
using LinearAlgebra

@testset "global_field_ranges partitions handler.total_dofs" begin
    mesh = create_unit_cube_mesh(Hex8; nx = 1, ny = 1, nz = 1)
    S = @DOFSet{T::DOF{Float64, Vertex}, u::DOF{Displacement{3}, Vertex}}
    _, h = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    rT, ru = global_field_ranges(h)
    nn = length(mesh.nodes)
    @test rT == 1:nn
    @test ru == (nn + 1):(nn + 3 * nn)
    @test rT.start == 1
    @test ru.stop == h.total_dofs
end

@testset "saddle_point_blocks dense views" begin
    K = [10.0 3.0 1.0; 3.0 11.0 2.0; 1.0 2.0 7.0]
    b = saddle_point_blocks(K, 1:2, 3:3)
    @test b.A ≈ K[1:2, 1:2]
    @test vec(b.B) ≈ [1.0, 2.0]
    @test vec(b.Bt) ≈ [1.0, 2.0]
    @test b.C[] ≈ 7.0
end
