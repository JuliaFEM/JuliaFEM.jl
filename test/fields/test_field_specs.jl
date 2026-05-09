# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

using Test
using JuliaFEM
using Tensors

@testset "DOFSet / field spec helpers" begin
    S = @DOFSet{T::DOF{Temperature, Vertex}, u::DOF{Displacement{3}, Vertex}}
    @test field_names(S) === (:T, :u)
    @test field_count(S) == 2
    @test !is_single_field(S)

    S1 = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    @test is_single_field(S1)

    @test field_ndofs(DOF{Float64, Vertex}, Tetrahedron{4}) == 4
    @test ndofs(DOF{Float64, Vertex}, Tetrahedron{4}) == 4
    @test ndofs(S, Tetrahedron{4}) == 4 + 12

    @test quantity_type(DOF{Float64, Vertex}) === Float64
    @test entity_type(DOF{Displacement{3}, Vertex}) === Vertex

    Bad = @NamedTuple{x::Int}
    @test_throws ErrorException ndofs(Bad, Tetrahedron{4})

    @test_throws ErrorException quantity_type(DOF{String, Vertex})
end
