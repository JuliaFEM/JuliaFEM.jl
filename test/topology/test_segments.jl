# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using Test
using StaticArrays

@testset "Segment Elements" begin
    @testset "Segment{2}" begin
        @test nnodes(Segment{2}()) == 2
        @test dim(Segment{2}()) == 1
        @test nvertices(Segment{2}()) == 2
        @test nedges(Segment{2}()) == 1
        @test nfaces(Segment{2}()) == 2  # Endpoints
        
        # Entity queries
        @test nentities(Segment{2}, Vertex) == 2
        @test nentities(Segment{2}, Edge) == 1
        @test nentities(Segment{2}, Cell) == 1
    end
    
    @testset "Segment{3}" begin
        @test nnodes(Segment{3}()) == 3
        @test nvertices(Segment{3}()) == 2  # Same topology as Segment{2}
        @test nedges(Segment{3}()) == 1
        @test nentities(Segment{3}, Edge) == 1
    end
end
