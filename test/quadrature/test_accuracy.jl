# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

@testset "Quadrature Accuracy Verification" begin

    @testset "1D Segment - Polynomial Integration" begin
        # Test that N-point rule integrates polynomials of degree 2N-1 exactly

        @testset "Order 1: Integrate constant (degree 0)" begin
            points = get_quadrature_points(Segment, GaussLegendre{1}())
            # ∫₋₁¹ 1 dx = 2
            result = sum(p.weight * 1.0 for p in points)
            @test result ≈ 2.0
        end

        @testset "Order 2: Integrate x³ (degree 3)" begin
            points = get_quadrature_points(Segment, GaussLegendre{2}())
            # ∫₋₁¹ x³ dx = 0 (odd function)
            result = sum(p.weight * p.coords[1]^3 for p in points)
            @test result ≈ 0.0 atol=1e-15

            # ∫₋₁¹ x² dx = 2/3
            result = sum(p.weight * p.coords[1]^2 for p in points)
            @test result ≈ 2/3 rtol=1e-10
        end

        @testset "Order 3: Integrate x⁵ (degree 5)" begin
            points = get_quadrature_points(Segment, GaussLegendre{3}())
            # ∫₋₁¹ x⁵ dx = 0 (odd function)
            result = sum(p.weight * p.coords[1]^5 for p in points)
            @test result ≈ 0.0 atol=1e-15

            # ∫₋₁¹ x⁴ dx = 2/5
            result = sum(p.weight * p.coords[1]^4 for p in points)
            @test result ≈ 2/5 rtol=1e-10
        end
    end

    @testset "2D Triangle - Polynomial Integration" begin
        # Reference triangle: vertices at (0,0), (1,0), (0,1)
        # Area = 0.5

        @testset "Order 1: Integrate constant" begin
            points = get_quadrature_points(Triangle, GaussLegendre{1}())
            # ∫∫ 1 dA = 0.5
            result = sum(p.weight * 1.0 for p in points)
            @test result ≈ 0.5
        end

        @testset "Order 2: Integrate linear functions" begin
            points = get_quadrature_points(Triangle, GaussLegendre{2}())

            # ∫∫ x dA = ∫₀¹ ∫₀^(1-x) x dy dx = 1/6
            result = sum(p.weight * p.coords[1] for p in points)
            @test result ≈ 1/6 rtol=1e-10

            # ∫∫ y dA = 1/6 (by symmetry)
            result = sum(p.weight * p.coords[2] for p in points)
            @test result ≈ 1/6 rtol=1e-10

            # ∫∫ (x+y) dA = 1/3
            result = sum(p.weight * (p.coords[1] + p.coords[2]) for p in points)
            @test result ≈ 1/3 rtol=1e-10
        end

        @testset "Order 3: Integrate quadratic functions" begin
            points = get_quadrature_points(Triangle, GaussLegendre{3}())

            # ∫∫ x² dA = ∫₀¹ ∫₀^(1-x) x² dy dx = 1/12
            result = sum(p.weight * p.coords[1]^2 for p in points)
            @test result ≈ 1/12 rtol=1e-10

            # ∫∫ xy dA = 1/24
            result = sum(p.weight * p.coords[1] * p.coords[2] for p in points)
            @test result ≈ 1/24 rtol=1e-10
        end
    end

    @testset "2D Quadrilateral - Polynomial Integration" begin
        # Reference quad: [-1,1]², Area = 4

        @testset "Order 2: Integrate x²" begin
            points = get_quadrature_points(Quadrilateral, GaussLegendre{2}())
            # ∫₋₁¹ ∫₋₁¹ x² dy dx = 2 * (2/3) * 2 = 8/3
            result = sum(p.weight * p.coords[1]^2 for p in points)
            @test result ≈ 8/3 rtol=1e-10
        end

        @testset "Order 3: Integrate x²y²" begin
            points = get_quadrature_points(Quadrilateral, GaussLegendre{3}())
            # ∫₋₁¹ ∫₋₁¹ x²y² dy dx = (2/3) * (2/3) = 4/9
            result = sum(p.weight * p.coords[1]^2 * p.coords[2]^2 for p in points)
            @test result ≈ 4/9 rtol=1e-10
        end
    end

    @testset "3D Tetrahedron - Polynomial Integration" begin
        # Reference tetrahedron: vertices at (0,0,0), (1,0,0), (0,1,0), (0,0,1)
        # Volume = 1/6

        @testset "Order 1: Integrate constant" begin
            points = get_quadrature_points(Tetrahedron, GaussLegendre{1}())
            # ∫∫∫ 1 dV = 1/6
            result = sum(p.weight * 1.0 for p in points)
            @test result ≈ 1/6
        end

        @testset "Order 2: Integrate linear functions" begin
            points = get_quadrature_points(Tetrahedron, GaussLegendre{2}())

            # ∫∫∫ x dV = 1/24 (by symmetry and integration)
            result = sum(p.weight * p.coords[1] for p in points)
            @test result ≈ 1/24 rtol=1e-10

            # ∫∫∫ (x+y+z) dV = 3/24 = 1/8
            result = sum(p.weight * (p.coords[1] + p.coords[2] + p.coords[3]) for p in points)
            @test result ≈ 1/8 rtol=1e-10
        end
    end

    @testset "3D Hexahedron - Polynomial Integration" begin
        # Reference hex: [-1,1]³, Volume = 8

        @testset "Order 2: Integrate x²" begin
            points = get_quadrature_points(Hexahedron, GaussLegendre{2}())
            # ∫₋₁¹ ∫₋₁¹ ∫₋₁¹ x² dz dy dx = 2 * 2 * (2/3) = 8/3
            result = sum(p.weight * p.coords[1]^2 for p in points)
            @test result ≈ 8/3 rtol=1e-10
        end

        @testset "Order 3: Integrate x²y²z²" begin
            points = get_quadrature_points(Hexahedron, GaussLegendre{3}())
            # ∫₋₁¹ ∫₋₁¹ ∫₋₁¹ x²y²z² dz dy dx = (2/3)³ = 8/27
            result = sum(p.weight * p.coords[1]^2 * p.coords[2]^2 * p.coords[3]^2 for p in points)
            @test result ≈ 8/27 rtol=1e-10
        end
    end

    @testset "Verify exactness limits" begin
        # GaussLegendre{N} should NOT be exact for degree 2N or higher

        @testset "Segment: Order 2 fails for degree 4" begin
            points = get_quadrature_points(Segment, GaussLegendre{2}())
            # ∫₋₁¹ x⁴ dx = 2/5, but 2-point rule gives different answer
            result = sum(p.weight * p.coords[1]^4 for p in points)
            exact = 2/5
            # Should have some error (not exact)
            @test abs(result - exact) > 1e-10
        end
    end
end
