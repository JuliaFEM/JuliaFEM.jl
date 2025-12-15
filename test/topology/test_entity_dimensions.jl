# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using Test

@testset "Entity Dimensions" begin
    # 1D elements
    @test dim(Segment{2}()) == 1
    @test dim(Segment{3}()) == 1
    
    # 2D elements
    @test dim(Triangle{3}()) == 2
    @test dim(Triangle{6}()) == 2
    @test dim(Quadrilateral{4}()) == 2
    @test dim(Quadrilateral{9}()) == 2
    
    # 3D elements
    @test dim(Tetrahedron{4}()) == 3
    @test dim(Tetrahedron{10}()) == 3
    @test dim(Hexahedron{8}()) == 3
    @test dim(Hexahedron{27}()) == 3
    @test dim(Pyramid{5}()) == 3
    @test dim(Wedge{6}()) == 3
    @test dim(Wedge{15}()) == 3
end
