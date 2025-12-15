# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using Test
using StaticArrays

@testset "Quadrilateral Elements" begin
    @testset "Quadrilateral{4}" begin
        @test nnodes(Quadrilateral{4}()) == 4
        @test dim(Quadrilateral{4}()) == 2
        @test nvertices(Quadrilateral{4}()) == 4
        @test nedges(Quadrilateral{4}()) == 4
        @test nfaces(Quadrilateral{4}()) == 1
        
        # Entity counts
        @test nentities(Quadrilateral{4}, Vertex) == 4
        @test nentities(Quadrilateral{4}, Edge) == 4
        @test nentities(Quadrilateral{4}, Face) == 1
        @test nentities(Quadrilateral{4}, Cell) == 1
    end
    
    @testset "Quadrilateral{8}" begin
        @test nnodes(Quadrilateral{8}()) == 8
        @test nvertices(Quadrilateral{8}()) == 4  # Same topology
        @test nedges(Quadrilateral{8}()) == 4
        @test nentities(Quadrilateral{8}, Edge) == 4
    end
    
    @testset "Quadrilateral{9}" begin
        @test nnodes(Quadrilateral{9}()) == 9
        @test nvertices(Quadrilateral{9}()) == 4  # Same topology
        @test nedges(Quadrilateral{9}()) == 4
        @test nentities(Quadrilateral{9}, Edge) == 4
    end
end
