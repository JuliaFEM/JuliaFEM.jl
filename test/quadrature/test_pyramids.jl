# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

@testset "Pyramid Quadrature Rules" begin

    @testset "GaussLegendre{2} - 5 points (default)" begin
        points = get_quadrature_points(Pyramid, GaussLegendre{2}())
        @test length(points) == 5

        # Check weights sum to volume of reference pyramid
        # Reference pyramid with base [-1,1]² and height from z=0 to z=1
        # Volume = (1/3) * base_area * height = (1/3) * 4 * 1 = 4/3
        weight_sum = sum(p.weight for p in points)
        @test weight_sum ≈ 4/3 rtol=1e-10

        # Check that there are 4 base points and 1 elevated point
        # (This is topology of typical pyramid quadrature)
        z_coords = [p.coords[3] for p in points]
        @test length(unique(z_coords)) >= 2  # At least 2 different z-levels
    end

    @testset "GaussLegendre{2,:B} - 5 points (variant B)" begin
        points = get_quadrature_points(Pyramid, GaussLegendre{2,:B}())
        @test length(points) == 5

        # Check weights sum to 4/3
        weight_sum = sum(p.weight for p in points)
        @test weight_sum ≈ 4/3 rtol=1e-10

        # All weights equal for variant B
        @test all(p.weight ≈ 2/15 for p in points)
    end

    @testset "Point coordinates are Vec type" begin
        points = get_quadrature_points(Pyramid, GaussLegendre{2}())
        @test points[1].coords isa Vec{3,Float64}
        @test !(points[1].coords isa SVector)
    end
end
