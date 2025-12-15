# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

@testset "Tensor Product Quadrature Rules" begin

    @testset "Segment (1D)" begin
        @testset "GaussLegendre{1} - 1 point" begin
            points = get_quadrature_points(Segment, GaussLegendre{1}())
            @test length(points) == 1
            @test points[1].coords[1] ≈ 0.0
            @test points[1].weight ≈ 2.0  # Length of [-1,1]
        end

        @testset "GaussLegendre{2} - 2 points" begin
            points = get_quadrature_points(Segment, GaussLegendre{2}())
            @test length(points) == 2

            # Check symmetric about origin
            @test points[1].coords[1] ≈ -1/sqrt(3)
            @test points[2].coords[1] ≈ 1/sqrt(3)

            # Check weights sum to 2
            @test sum(p.weight for p in points) ≈ 2.0
        end

        @testset "GaussLegendre{3} - 3 points" begin
            points = get_quadrature_points(Segment, GaussLegendre{3}())
            @test length(points) == 3

            # Middle point at origin
            @test points[2].coords[1] ≈ 0.0

            # Check weights sum to 2
            @test sum(p.weight for p in points) ≈ 2.0
        end
    end

    @testset "Quadrilateral (2D)" begin
        @testset "GaussLegendre{1} - 1×1 = 1 point" begin
            points = get_quadrature_points(Quadrilateral, GaussLegendre{1}())
            @test length(points) == 1
            @test points[1].coords[1] ≈ 0.0
            @test points[1].coords[2] ≈ 0.0
            @test points[1].weight ≈ 4.0  # Area of [-1,1]²
        end

        @testset "GaussLegendre{2} - 2×2 = 4 points" begin
            points = get_quadrature_points(Quadrilateral, GaussLegendre{2}())
            @test length(points) == 4

            # Check weights sum to 4
            @test sum(p.weight for p in points) ≈ 4.0

            # Check all weights are equal
            @test all(p.weight ≈ 1.0 for p in points)

            # Check symmetry
            a = 1/sqrt(3)
            expected_coords = [
                Vec{2}(-a, -a), Vec{2}(a, -a),
                Vec{2}(-a, a), Vec{2}(a, a)
            ]
            for ec in expected_coords
                @test any(p -> p.coords[1] ≈ ec[1] && p.coords[2] ≈ ec[2], points)
            end
        end

        @testset "GaussLegendre{3} - 3×3 = 9 points" begin
            points = get_quadrature_points(Quadrilateral, GaussLegendre{3}())
            @test length(points) == 9

            # Check weights sum to 4
            @test sum(p.weight for p in points) ≈ 4.0

            # Center point should exist
            @test any(p -> p.coords[1] ≈ 0.0 && p.coords[2] ≈ 0.0, points)
        end

        @testset "GaussLegendre{4} - 4×4 = 16 points" begin
            points = get_quadrature_points(Quadrilateral, GaussLegendre{4}())
            @test length(points) == 16
            @test sum(p.weight for p in points) ≈ 4.0 rtol=1e-10
        end

        @testset "GaussLegendre{5} - 5×5 = 25 points" begin
            points = get_quadrature_points(Quadrilateral, GaussLegendre{5}())
            @test length(points) == 25
            @test sum(p.weight for p in points) ≈ 4.0 rtol=1e-10
        end
    end

    @testset "Hexahedron (3D)" begin
        @testset "GaussLegendre{1} - 1×1×1 = 1 point" begin
            points = get_quadrature_points(Hexahedron, GaussLegendre{1}())
            @test length(points) == 1
            @test points[1].coords[1] ≈ 0.0
            @test points[1].coords[2] ≈ 0.0
            @test points[1].coords[3] ≈ 0.0
            @test points[1].weight ≈ 8.0  # Volume of [-1,1]³
        end

        @testset "GaussLegendre{2} - 2×2×2 = 8 points" begin
            points = get_quadrature_points(Hexahedron, GaussLegendre{2}())
            @test length(points) == 8

            # Check weights sum to 8
            @test sum(p.weight for p in points) ≈ 8.0

            # Check all weights are equal
            @test all(p.weight ≈ 1.0 for p in points)

            # Check corners of [-a,a]³ cube
            a = 1/sqrt(3)
            @test all(abs(p.coords[1]) ≈ a for p in points)
            @test all(abs(p.coords[2]) ≈ a for p in points)
            @test all(abs(p.coords[3]) ≈ a for p in points)
        end

        @testset "GaussLegendre{3} - 3×3×3 = 27 points" begin
            points = get_quadrature_points(Hexahedron, GaussLegendre{3}())
            @test length(points) == 27

            # Check weights sum to 8
            @test sum(p.weight for p in points) ≈ 8.0

            # Center point should exist
            @test any(p -> p.coords[1] ≈ 0.0 && p.coords[2] ≈ 0.0 && p.coords[3] ≈ 0.0, points)
        end

        @testset "GaussLegendre{4} - 4×4×4 = 64 points" begin
            points = get_quadrature_points(Hexahedron, GaussLegendre{4}())
            @test length(points) == 64
            @test sum(p.weight for p in points) ≈ 8.0 rtol=1e-10
        end

        @testset "GaussLegendre{5} - 5×5×5 = 125 points" begin
            points = get_quadrature_points(Hexahedron, GaussLegendre{5}())
            @test length(points) == 125
            @test sum(p.weight for p in points) ≈ 8.0 rtol=1e-10
        end
    end

    @testset "Point coordinates are Vec type" begin
        points_seg = get_quadrature_points(Segment, GaussLegendre{2}())
        @test points_seg[1].coords isa Vec{1,Float64}

        points_quad = get_quadrature_points(Quadrilateral, GaussLegendre{2}())
        @test points_quad[1].coords isa Vec{2,Float64}

        points_hex = get_quadrature_points(Hexahedron, GaussLegendre{2}())
        @test points_hex[1].coords isa Vec{3,Float64}
    end
end
