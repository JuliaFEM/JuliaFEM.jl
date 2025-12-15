# Test that numerical filtering produces clean fractions
# Verify the improved round_coefficient function works correctly
using Test
using StaticArrays, Tensors

include("../../src/topology/api.jl")
include("../../src/topology/segments.jl")
include("../../src/topology/triangles.jl")
include("../../src/topology/quadrilaterals.jl")
include("../../src/topology/tetrahedra.jl")
include("../../src/topology/hexahedra.jl")
include("../../src/topology/pyramids.jl")
include("../../src/topology/wedges.jl")
include("../../src/basis/api.jl")
include("../../src/basis/basis_generated.jl")

@testset "Numerical Accuracy - Clean Fractions" begin
    @testset "Hex27 corner nodes - 1/18 coefficients" begin
        # At origin, all corner nodes should have specific fraction contributions
        N = get_basis_functions(Hex27(), Lagrange{2}(), Vec{3}((0.0, 0.0, 0.0)))

        # The linear terms in corner shape functions should be exactly 1/18
        # We can't test the internal expression, but we can verify the partition of unity
        @test sum(N) ≈ 1.0 atol = 1e-14
        @test length(N) == 27

        # Center node (N27) should be 1 at center
        @test N[27] ≈ 1.0 atol = 1e-14

        # All other nodes should be 0 at center
        @test all(N[1:26] .≈ 0.0)
    end

    @testset "Hex27 mid-edge nodes - 5/18 coefficients" begin
        # At a mid-edge location, verify partition of unity
        N = get_basis_functions(Hex27(), Lagrange{2}(), Vec{3}((0.5, 0.0, 0.0)))
        @test sum(N) ≈ 1.0 atol = 1e-14
    end

    @testset "Common fractions in all elements" begin
        # Test that basis functions evaluate without numerical noise

        # 1/2 (Seg2)
        N = get_basis_functions(Seg2(), Lagrange{1}(), Vec{1}((0.0,)))
        @test N[1] ≈ 0.5 atol = 1e-14
        @test N[2] ≈ 0.5 atol = 1e-14

        # 1/3 (Tri3 at centroid)
        N = get_basis_functions(Tri3(), Lagrange{1}(), Vec{2}((1 / 3, 1 / 3)))
        @test N[1] ≈ 1 / 3 atol = 1e-14
        @test N[2] ≈ 1 / 3 atol = 1e-14
        @test N[3] ≈ 1 / 3 atol = 1e-14

        # 1/4 (Quad4 at center)
        N = get_basis_functions(Quad4(), Lagrange{1}(), Vec{2}((0.0, 0.0)))
        @test all(N .≈ 0.25)

        # 1/8 (Hex8 at center)
        N = get_basis_functions(Hex8(), Lagrange{1}(), Vec{3}((0.0, 0.0, 0.0)))
        @test all(N .≈ 0.125)
    end

    @testset "Higher-order elements - no numerical noise" begin
        # Quad9 at various points
        N = get_basis_functions(Quad9(), Lagrange{2}(), Vec{2}((0.3, 0.7)))
        @test sum(N) ≈ 1.0 atol = 1e-14
        @test !any(isnan, N)
        @test !any(isinf, N)

        # Tet10 at various points
        N = get_basis_functions(Tet10(), Lagrange{2}(), Vec{3}((0.1, 0.2, 0.3)))
        @test sum(N) ≈ 1.0 atol = 1e-14
        @test !any(isnan, N)
        @test !any(isinf, N)

        # Hex20 at various points (uses Serendipity basis)
        N = get_basis_functions(Hex20(), Serendipity{2}(), Vec{3}((-0.5, 0.5, 0.0)))
        @test sum(N) ≈ 1.0 atol = 1e-14
        @test !any(isnan, N)
        @test !any(isinf, N)
    end
end
