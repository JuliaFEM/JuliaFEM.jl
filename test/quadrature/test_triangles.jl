# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

@testset "Triangle Quadrature Rules" begin

    @testset "GaussLegendre{1} - 1 point" begin
        points = get_quadrature_points(Triangle, GaussLegendre{1}())
        @test length(points) == 1

        # Check centroid location
        @test points[1].coords[1] ≈ 1/3
        @test points[1].coords[2] ≈ 1/3

        # Check weight sums to area of reference triangle (0.5)
        @test points[1].weight ≈ 0.5
    end

    @testset "GaussLegendre{2} - 3 points (default)" begin
        points = get_quadrature_points(Triangle, GaussLegendre{2}())
        @test length(points) == 3

        # Check weights sum to 0.5
        weight_sum = sum(p.weight for p in points)
        @test weight_sum ≈ 0.5

        # Check all points are inside triangle
        for p in points
            ξ, η = p.coords[1], p.coords[2]
            @test ξ >= 0 && η >= 0 && ξ + η <= 1
        end
    end

    @testset "GaussLegendre{2,:B} - 3 points (variant B)" begin
        points = get_quadrature_points(Triangle, GaussLegendre{2,:B}())
        @test length(points) == 3

        # Check weights sum to 0.5
        weight_sum = sum(p.weight for p in points)
        @test weight_sum ≈ 0.5

        # Variant B uses edge midpoints
        @test any(p -> p.coords[1] ≈ 0.5 && p.coords[2] ≈ 0.0, points)
        @test any(p -> p.coords[1] ≈ 0.0 && p.coords[2] ≈ 0.5, points)
        @test any(p -> p.coords[1] ≈ 0.5 && p.coords[2] ≈ 0.5, points)
    end

    @testset "GaussLegendre{3} - 4 points (default)" begin
        points = get_quadrature_points(Triangle, GaussLegendre{3}())
        @test length(points) == 4

        # Check weights sum to 0.5
        weight_sum = sum(p.weight for p in points)
        @test weight_sum ≈ 0.5

        # All weights should be positive for default variant
        @test all(p.weight > 0 for p in points)
    end

    @testset "GaussLegendre{3,:B} - 4 points (variant B, has negative weight)" begin
        points = get_quadrature_points(Triangle, GaussLegendre{3,:B}())
        @test length(points) == 4

        # Check weights sum to 0.5
        weight_sum = sum(p.weight for p in points)
        @test weight_sum ≈ 0.5

        # This variant has one negative weight
        @test any(p.weight < 0 for p in points)
    end

    @testset "GaussLegendre{4} - 6 points" begin
        points = get_quadrature_points(Triangle, GaussLegendre{4}())
        @test length(points) == 6

        # Check weights sum to 0.5
        weight_sum = sum(p.weight for p in points)
        @test weight_sum ≈ 0.5 rtol=1e-10
    end

    @testset "GaussLegendre{5} - 7 points" begin
        points = get_quadrature_points(Triangle, GaussLegendre{5}())
        @test length(points) == 7

        # Check weights sum to 0.5
        weight_sum = sum(p.weight for p in points)
        @test weight_sum ≈ 0.5 rtol=1e-10

        # Should include centroid point
        @test any(p -> p.coords[1] ≈ 1/3 && p.coords[2] ≈ 1/3, points)
    end

    @testset "GaussLegendre{6} - 12 points" begin
        points = get_quadrature_points(Triangle, GaussLegendre{6}())
        @test length(points) == 12

        # Check weights sum to 0.5
        weight_sum = sum(p.weight for p in points)
        @test weight_sum ≈ 0.5 rtol=1e-10
    end

    @testset "Point coordinates are Vec type" begin
        points = get_quadrature_points(Triangle, GaussLegendre{2}())
        @test points[1].coords isa Vec{2,Float64}
        @test !(points[1].coords isa SVector)
    end
end
