# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

@testset "Wedge Quadrature Rules" begin

    @testset "GaussLegendre{2} - 6 points (default)" begin
        points = get_quadrature_points(Wedge, GaussLegendre{2}())
        @test length(points) == 6

        # Check weights sum to 1 (volume of reference wedge)
        weight_sum = sum(p.weight for p in points)
        @test weight_sum ≈ 1.0

        # All weights should be equal
        @test all(p.weight ≈ 1/6 for p in points)

        # Check z coordinates are at ±1/sqrt(3)
        a = 1/sqrt(3)
        @test sum(abs(p.coords[3]) ≈ a for p in points) == 6
    end

    @testset "GaussLegendre{2,:B} - 6 points (variant B)" begin
        points = get_quadrature_points(Wedge, GaussLegendre{2,:B}())
        @test length(points) == 6

        # Check weights sum to 1
        weight_sum = sum(p.weight for p in points)
        @test weight_sum ≈ 1.0

        # All weights should be equal
        @test all(p.weight ≈ 1/6 for p in points)
    end

    @testset "GaussLegendre{5} - 21 points" begin
        points = get_quadrature_points(Wedge, GaussLegendre{5}())
        @test length(points) == 21

        # Check weights sum to 1
        weight_sum = sum(p.weight for p in points)
        @test weight_sum ≈ 1.0 rtol=1e-10

        # Should have 3 layers (z = -alpha, 0, alpha)
        a = sqrt(3/5)
        z_values = [p.coords[3] for p in points]
        @test sum(abs(z) ≈ a for z in z_values) == 14  # 7 points at each ±alpha
        @test sum(abs(z) ≈ 0.0 for z in z_values) == 7   # 7 points at z=0
    end

    @testset "Point coordinates are Vec type" begin
        points = get_quadrature_points(Wedge, GaussLegendre{2}())
        @test points[1].coords isa Vec{3,Float64}
        @test !(points[1].coords isa SVector)
    end
end
