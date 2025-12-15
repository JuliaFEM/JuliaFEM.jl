# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using Test
using StaticArrays

@testset "Pyramid Elements" begin
    @testset "Pyramid{5}" begin
        @test nnodes(Pyramid{5}()) == 5
        @test dim(Pyramid{5}()) == 3
        @test nvertices(Pyramid{5}()) == 5
        @test nedges(Pyramid{5}()) == 8
        @test nfaces(Pyramid{5}()) == 5
        
        # Entity counts
        @test nentities(Pyramid{5}, Vertex) == 5
        @test nentities(Pyramid{5}, Edge) == 8
        @test nentities(Pyramid{5}, Face) == 5
        @test nentities(Pyramid{5}, Cell) == 1
    end
end
