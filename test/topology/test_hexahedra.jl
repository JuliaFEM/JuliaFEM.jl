# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using Test
using StaticArrays

@testset "Hexahedron Elements" begin
    @testset "Hexahedron{8}" begin
        @test nnodes(Hexahedron{8}()) == 8
        @test dim(Hexahedron{8}()) == 3
        @test nvertices(Hexahedron{8}()) == 8
        @test nedges(Hexahedron{8}()) == 12
        @test nfaces(Hexahedron{8}()) == 6
        
        # Entity counts
        @test nentities(Hexahedron{8}, Vertex) == 8
        @test nentities(Hexahedron{8}, Edge) == 12
        @test nentities(Hexahedron{8}, Face) == 6
        @test nentities(Hexahedron{8}, Cell) == 1
    end
    
    @testset "Hexahedron{20}" begin
        @test nnodes(Hexahedron{20}()) == 20
        @test nvertices(Hexahedron{20}()) == 8  # Same topology
        @test nedges(Hexahedron{20}()) == 12
        @test nfaces(Hexahedron{20}()) == 6
        @test nentities(Hexahedron{20}, Edge) == 12
    end
    
    @testset "Hexahedron{27}" begin
        @test nnodes(Hexahedron{27}()) == 27
        @test nvertices(Hexahedron{27}()) == 8  # Same topology
        @test nedges(Hexahedron{27}()) == 12
        @test nfaces(Hexahedron{27}()) == 6
        @test nentities(Hexahedron{27}, Edge) == 12
    end
end
