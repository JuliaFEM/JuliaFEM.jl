# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using Test

@testset "Topological Invariance" begin
    # Linear and quadratic variants have same topology
    
    @testset "Segments" begin
        @test nvertices(Segment{2}()) == nvertices(Segment{3}())
        @test nedges(Segment{2}()) == nedges(Segment{3}())
    end
    
    @testset "Triangles" begin
        @test nvertices(Triangle{3}()) == nvertices(Triangle{6}()) == nvertices(Triangle{7}()) == nvertices(Triangle{10}())
        @test nedges(Triangle{3}()) == nedges(Triangle{6}()) == nedges(Triangle{7}()) == nedges(Triangle{10}())
        @test nfaces(Triangle{3}()) == nfaces(Triangle{6}()) == nfaces(Triangle{7}()) == nfaces(Triangle{10}())
    end
    
    @testset "Quadrilaterals" begin
        @test nvertices(Quadrilateral{4}()) == nvertices(Quadrilateral{8}()) == nvertices(Quadrilateral{9}())
        @test nedges(Quadrilateral{4}()) == nedges(Quadrilateral{8}()) == nedges(Quadrilateral{9}())
        @test nfaces(Quadrilateral{4}()) == nfaces(Quadrilateral{8}()) == nfaces(Quadrilateral{9}())
    end
    
    @testset "Tetrahedra" begin
        @test nvertices(Tetrahedron{4}()) == nvertices(Tetrahedron{10}())
        @test nedges(Tetrahedron{4}()) == nedges(Tetrahedron{10}())
        @test nfaces(Tetrahedron{4}()) == nfaces(Tetrahedron{10}())
    end
    
    @testset "Hexahedra" begin
        @test nvertices(Hexahedron{8}()) == nvertices(Hexahedron{20}()) == nvertices(Hexahedron{27}())
        @test nedges(Hexahedron{8}()) == nedges(Hexahedron{20}()) == nedges(Hexahedron{27}())
        @test nfaces(Hexahedron{8}()) == nfaces(Hexahedron{20}()) == nfaces(Hexahedron{27}())
    end
    
    @testset "Wedges" begin
        @test nvertices(Wedge{6}()) == nvertices(Wedge{15}())
        @test nedges(Wedge{6}()) == nedges(Wedge{15}())
        @test nfaces(Wedge{6}()) == nfaces(Wedge{15}())
    end
end
