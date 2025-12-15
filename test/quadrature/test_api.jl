# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

@testset "API Types and Interface" begin

    @testset "QuadraturePoint construction" begin
        # Test 1D point
        qp1 = QuadraturePoint(Vec{1}(0.5), 1.0)
        @test qp1.coords isa Vec{1,Float64}
        @test qp1.coords[1] == 0.5
        @test qp1.weight == 1.0

        # Test 2D point
        qp2 = QuadraturePoint(Vec{2}(0.5, 0.25), 0.5)
        @test qp2.coords isa Vec{2,Float64}
        @test qp2.coords[1] == 0.5
        @test qp2.coords[2] == 0.25
        @test qp2.weight == 0.5

        # Test 3D point
        qp3 = QuadraturePoint(Vec{3}(0.5, 0.25, 0.125), 0.25)
        @test qp3.coords isa Vec{3,Float64}
        @test qp3.coords[1] == 0.5
        @test qp3.coords[2] == 0.25
        @test qp3.coords[3] == 0.125
        @test qp3.weight == 0.25
    end

    @testset "GaussLegendre type" begin
        gl1 = GaussLegendre{1}()
        @test gl1 isa AbstractQuadratureRule

        gl2 = GaussLegendre{2}()
        @test gl2 isa AbstractQuadratureRule

        gl2b = GaussLegendre{2,:B}()
        @test gl2b isa AbstractQuadratureRule
    end

    @testset "get_quadrature_points return type" begin
        # Test that return type is SVector
        points_tri = get_quadrature_points(Triangle, GaussLegendre{2}())
        @test points_tri isa SVector
        @test eltype(points_tri) <: QuadraturePoint{2}
        @test length(points_tri) == 3

        points_hex = get_quadrature_points(Hexahedron, GaussLegendre{2}())
        @test points_hex isa SVector
        @test eltype(points_hex) <: QuadraturePoint{3}
        @test length(points_hex) == 8
    end

    @testset "default_quadrature" begin
        # Level 1: From basis order only
        rule1 = default_quadrature(1)
        @test rule1 isa GaussLegendre{2}

        rule2 = default_quadrature(2)
        @test rule2 isa GaussLegendre{3}

        # Level 2: From topology + basis order
        rule_hex = default_quadrature(Hexahedron, 1)
        @test rule_hex isa GaussLegendre{2}
    end

    @testset "npoints function" begin
        # Test that npoints returns correct count
        @test npoints(Triangle, GaussLegendre{1}()) == 1
        @test npoints(Triangle, GaussLegendre{2}()) == 3
        @test npoints(Triangle, GaussLegendre{3}()) == 4

        @test npoints(Hexahedron, GaussLegendre{1}()) == 1
        @test npoints(Hexahedron, GaussLegendre{2}()) == 8
        @test npoints(Hexahedron, GaussLegendre{3}()) == 27

        @test npoints(Quadrilateral, GaussLegendre{2}()) == 4
        @test npoints(Quadrilateral, GaussLegendre{3}()) == 9
    end
end
