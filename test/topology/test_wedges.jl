# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using Test
using StaticArrays

@testset "Wedge Elements" begin
    @testset "Wedge{6}" begin
        @test nnodes(Wedge{6}()) == 6
        @test dim(Wedge{6}()) == 3
        @test nvertices(Wedge{6}()) == 6
        @test nedges(Wedge{6}()) == 9
        @test nfaces(Wedge{6}()) == 5
        
        # Entity counts
        @test nentities(Wedge{6}, Vertex) == 6
        @test nentities(Wedge{6}, Edge) == 9
        @test nentities(Wedge{6}, Face) == 5
        @test nentities(Wedge{6}, Cell) == 1
    end
    
    @testset "Wedge{15}" begin
        @test nnodes(Wedge{15}()) == 15
        @test nvertices(Wedge{15}()) == 6  # Same topology
        @test nedges(Wedge{15}()) == 9
        @test nfaces(Wedge{15}()) == 5
        @test nentities(Wedge{15}, Edge) == 9
    end
end
