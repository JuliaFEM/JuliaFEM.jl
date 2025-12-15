# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

@testset "Legacy API Compatibility" begin

    @testset "Triangle legacy symbols" begin
        # Test that old Val{:GLTRI*} symbols still work
        @test get_quadrature_points(Val{:GLTRI1}) !== nothing
        @test get_quadrature_points(Val{:GLTRI3}) !== nothing
        @test get_quadrature_points(Val{:GLTRI3B}) !== nothing
        @test get_quadrature_points(Val{:GLTRI4}) !== nothing
        @test get_quadrature_points(Val{:GLTRI4B}) !== nothing
        @test get_quadrature_points(Val{:GLTRI6}) !== nothing
        @test get_quadrature_points(Val{:GLTRI7}) !== nothing
        @test get_quadrature_points(Val{:GLTRI12}) !== nothing

        # Test order queries
        @test get_order(Val{:GLTRI1}) == 1
        @test get_order(Val{:GLTRI3}) == 2
        @test get_order(Val{:GLTRI4}) == 3
    end

    @testset "Tetrahedron legacy symbols" begin
        @test get_quadrature_points(Val{:GLTET1}) !== nothing
        @test get_quadrature_points(Val{:GLTET4}) !== nothing
        @test get_quadrature_points(Val{:GLTET5}) !== nothing
        @test get_quadrature_points(Val{:GLTET15}) !== nothing

        @test get_order(Val{:GLTET1}) == 1
        @test get_order(Val{:GLTET4}) == 2
        @test get_order(Val{:GLTET5}) == 3
        @test get_order(Val{:GLTET15}) == 4
    end

    @testset "Wedge legacy symbols" begin
        @test get_quadrature_points(Val{:GLWED6}) !== nothing
        @test get_quadrature_points(Val{:GLWED6B}) !== nothing
        @test get_quadrature_points(Val{:GLWED21}) !== nothing

        @test get_order(Val{:GLWED6}) == 1
        @test get_order(Val{:GLWED6B}) == 1
        @test get_order(Val{:GLWED21}) == 5
    end

    @testset "Pyramid legacy symbols" begin
        @test get_quadrature_points(Val{:GLPYR5}) !== nothing
        @test get_quadrature_points(Val{:GLPYR5B}) !== nothing

        @test get_order(Val{:GLPYR5}) == 1
        @test get_order(Val{:GLPYR5B}) == 1
    end

    @testset "Tensor product legacy symbols" begin
        # Segments
        @test get_quadrature_points(Val{:GLSEG1}) !== nothing
        @test get_quadrature_points(Val{:GLSEG2}) !== nothing
        @test get_order(Val{:GLSEG1}) == 1
        @test get_order(Val{:GLSEG2}) == 3

        # Quadrilaterals
        @test get_quadrature_points(Val{:GLQUAD1}) !== nothing
        @test get_quadrature_points(Val{:GLQUAD4}) !== nothing
        @test get_quadrature_points(Val{:GLQUAD9}) !== nothing
        @test get_order(Val{:GLQUAD1}) == 1
        @test get_order(Val{:GLQUAD4}) == 3

        # Hexahedra
        @test get_quadrature_points(Val{:GLHEX1}) !== nothing
        @test get_quadrature_points(Val{:GLHEX8}) !== nothing
        @test get_quadrature_points(Val{:GLHEX27}) !== nothing
        @test get_order(Val{:GLHEX1}) == 1
        @test get_order(Val{:GLHEX8}) == 3
    end
end
