# Test basis functions at element nodes
# At a node, the corresponding basis function should be 1, all others 0 (Kronecker delta property)
using Test
using StaticArrays, Tensors

using JuliaFEM

@testset "Kronecker Delta Property at Nodes" begin
    @testset "Seg2 at nodes" begin
        # Node 1: xi = -1
        N = get_basis_functions(Seg2(), Lagrange{1}(), Vec{1}((-1.0,)))
        @test N[1] ≈ 1.0 atol = 1e-14
        @test N[2] ≈ 0.0 atol = 1e-14

        # Node 2: xi = 1
        N = get_basis_functions(Seg2(), Lagrange{1}(), Vec{1}((1.0,)))
        @test N[1] ≈ 0.0 atol = 1e-14
        @test N[2] ≈ 1.0 atol = 1e-14
    end

    @testset "Tri3 at nodes" begin
        # Node 1: (0, 0)
        N = get_basis_functions(Tri3(), Lagrange{1}(), Vec{2}((0.0, 0.0)))
        @test N[1] ≈ 1.0 atol = 1e-14
        @test N[2] ≈ 0.0 atol = 1e-14
        @test N[3] ≈ 0.0 atol = 1e-14

        # Node 2: (1, 0)
        N = get_basis_functions(Tri3(), Lagrange{1}(), Vec{2}((1.0, 0.0)))
        @test N[1] ≈ 0.0 atol = 1e-14
        @test N[2] ≈ 1.0 atol = 1e-14
        @test N[3] ≈ 0.0 atol = 1e-14

        # Node 3: (0, 1)
        N = get_basis_functions(Tri3(), Lagrange{1}(), Vec{2}((0.0, 1.0)))
        @test N[1] ≈ 0.0 atol = 1e-14
        @test N[2] ≈ 0.0 atol = 1e-14
        @test N[3] ≈ 1.0 atol = 1e-14
    end

    @testset "Quad4 at nodes" begin
        # Node 1: (-1, -1)
        N = get_basis_functions(Quad4(), Lagrange{1}(), Vec{2}((-1.0, -1.0)))
        @test N[1] ≈ 1.0 atol = 1e-14
        @test N[2] ≈ 0.0 atol = 1e-14
        @test N[3] ≈ 0.0 atol = 1e-14
        @test N[4] ≈ 0.0 atol = 1e-14

        # Node 3: (1, 1)
        N = get_basis_functions(Quad4(), Lagrange{1}(), Vec{2}((1.0, 1.0)))
        @test N[1] ≈ 0.0 atol = 1e-14
        @test N[2] ≈ 0.0 atol = 1e-14
        @test N[3] ≈ 1.0 atol = 1e-14
        @test N[4] ≈ 0.0 atol = 1e-14
    end

    @testset "Tet4 at nodes" begin
        # Node 1: (0, 0, 0)
        N = get_basis_functions(Tet4(), Lagrange{1}(), Vec{3}((0.0, 0.0, 0.0)))
        @test N[1] ≈ 1.0 atol = 1e-14
        @test all(N[2:4] .≈ 0.0)

        # Node 2: (1, 0, 0)
        N = get_basis_functions(Tet4(), Lagrange{1}(), Vec{3}((1.0, 0.0, 0.0)))
        @test N[2] ≈ 1.0 atol = 1e-14
        @test N[1] ≈ 0.0 atol = 1e-14
        @test N[3] ≈ 0.0 atol = 1e-14
        @test N[4] ≈ 0.0 atol = 1e-14
    end

    @testset "Hex8 at nodes" begin
        # Node 1: (-1, -1, -1)
        N = get_basis_functions(Hex8(), Lagrange{1}(), Vec{3}((-1.0, -1.0, -1.0)))
        @test N[1] ≈ 1.0 atol=1e-14
        @test all(N[2:8] .≈ 0.0)

        # Check interior point for partition of unity
        N = get_basis_functions(Hex8(), Lagrange{1}(), Vec{3}((0.5, 0.3, -0.2)))
        @test sum(N) ≈ 1.0 atol=1e-14
    end
end
