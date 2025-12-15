# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

@testset "Tetrahedron Quadrature Rules" begin

    @testset "GaussLegendre{1} - 1 point (centroid)" begin
        points = get_quadrature_points(Tetrahedron, GaussLegendre{1}())
        @test length(points) == 1

        # Check centroid location
        @test points[1].coords[1] ≈ 1/4
        @test points[1].coords[2] ≈ 1/4
        @test points[1].coords[3] ≈ 1/4

        # Check weight equals volume of reference tetrahedron (1/6)
        @test points[1].weight ≈ 1/6
    end

    @testset "GaussLegendre{2} - 4 points" begin
        points = get_quadrature_points(Tetrahedron, GaussLegendre{2}())
        @test length(points) == 4

        # Check weights sum to 1/6
        weight_sum = sum(p.weight for p in points)
        @test weight_sum ≈ 1/6

        # Check all points are inside tetrahedron
        for p in points
            ξ, η, ζ = p.coords[1], p.coords[2], p.coords[3]
            @test ξ >= 0 && η >= 0 && ζ >= 0
            @test ξ + η + ζ <= 1
        end

        # All weights should be equal for this symmetric rule
        @test all(p.weight ≈ points[1].weight for p in points)
    end

    @testset "GaussLegendre{3} - 5 points (has negative weight)" begin
        points = get_quadrature_points(Tetrahedron, GaussLegendre{3}())
        @test length(points) == 5

        # Check weights sum to 1/6
        weight_sum = sum(p.weight for p in points)
        @test weight_sum ≈ 1/6

        # This rule has one negative weight at centroid
        @test any(p.weight < 0 for p in points)

        # Centroid point should have negative weight
        centroid_point = findfirst(p -> p.coords[1] ≈ 1/4 && p.coords[2] ≈ 1/4, points)
        @test points[centroid_point].weight < 0
    end

    @testset "GaussLegendre{4} - 15 points" begin
        points = get_quadrature_points(Tetrahedron, GaussLegendre{4}())
        @test length(points) == 15

        # Check weights sum to 1/6
        weight_sum = sum(p.weight for p in points)
        @test weight_sum ≈ 1/6 rtol=1e-10

        # All weights should be positive
        @test all(p.weight > 0 for p in points)

        # Check all points are inside tetrahedron
        for p in points
            ξ, η, ζ = p.coords[1], p.coords[2], p.coords[3]
            @test ξ >= 0 && η >= 0 && ζ >= 0
            @test ξ + η + ζ <= 1 + 1e-10  # Small tolerance for rounding
        end
    end

    @testset "Point coordinates are Vec type" begin
        points = get_quadrature_points(Tetrahedron, GaussLegendre{2}())
        @test points[1].coords isa Vec{3,Float64}
        @test !(points[1].coords isa SVector)
    end
end
