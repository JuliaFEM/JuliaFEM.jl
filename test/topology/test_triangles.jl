# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using Test
using StaticArrays

@testset "Triangle Elements" begin
    @testset "Triangle{3}" begin
        @test nnodes(Triangle{3}()) == 3
        @test dim(Triangle{3}()) == 2
        @test nvertices(Triangle{3}()) == 3
        @test nedges(Triangle{3}()) == 3
        @test nfaces(Triangle{3}()) == 1
        
        # Entity counts
        @test nentities(Triangle{3}, Vertex) == 3
        @test nentities(Triangle{3}, Edge) == 3
        @test nentities(Triangle{3}, Face) == 1
        @test nentities(Triangle{3}, Cell) == 1
    end
    
    @testset "Triangle{6}" begin
        @test nnodes(Triangle{6}()) == 6
        @test nvertices(Triangle{6}()) == 3  # Same topology
        @test nedges(Triangle{6}()) == 3
        @test nfaces(Triangle{6}()) == 1
        @test nentities(Triangle{6}, Edge) == 3
    end
    
    @testset "Triangle{7}" begin
        @test nnodes(Triangle{7}()) == 7
        @test nvertices(Triangle{7}()) == 3  # Same topology
        @test nedges(Triangle{7}()) == 3
        @test nentities(Triangle{7}, Edge) == 3
    end
    
    @testset "Triangle{10}" begin
        @test nnodes(Triangle{10}()) == 10
        @test nvertices(Triangle{10}()) == 3  # Same topology
        @test nedges(Triangle{10}()) == 3
        @test nentities(Triangle{10}, Edge) == 3
    end
end
