# Test gradient properties for all elements
using Test
using StaticArrays, Tensors

using JuliaFEM

@testset "Gradient Properties" begin
    @testset "Linear elements - constant gradients" begin
        # Tri3 gradients should be constant everywhere
        dN1 = get_basis_derivatives(Tri3(), Lagrange{1}(), Vec{2}((0.2, 0.3)))
        dN2 = get_basis_derivatives(Tri3(), Lagrange{1}(), Vec{2}((0.5, 0.4)))

        for i in 1:3
            @test dN1[i] ≈ dN2[i] atol = 1e-14
        end

        # Tet4 gradients should be constant everywhere
        dN1 = get_basis_derivatives(Tet4(), Lagrange{1}(), Vec{3}((0.1, 0.2, 0.3)))
        dN2 = get_basis_derivatives(Tet4(), Lagrange{1}(), Vec{3}((0.2, 0.1, 0.4)))

        for i in 1:4
            @test dN1[i] ≈ dN2[i] atol = 1e-14
        end
    end

    @testset "Sum of gradients = 0 (rigid body motion)" begin
        # For any basis, sum of all gradients should be zero vector
        # (ensures constant field has zero gradient)

        # 1D
        dN = get_basis_derivatives(Seg2(), Lagrange{1}(), Vec{1}((0.3,)))
        @test sum(dN)[1] ≈ 0.0 atol = 1e-14

        # 2D
        dN = get_basis_derivatives(Tri6(), Lagrange{2}(), Vec{2}((0.3, 0.4)))
        sum_grad = sum(dN)
        @test sum_grad[1] ≈ 0.0 atol = 1e-14
        @test sum_grad[2] ≈ 0.0 atol = 1e-14

        dN = get_basis_derivatives(Quad9(), Lagrange{2}(), Vec{2}((0.5, -0.5)))
        sum_grad = sum(dN)
        @test sum_grad[1] ≈ 0.0 atol = 1e-14
        @test sum_grad[2] ≈ 0.0 atol = 1e-14

        # 3D
        dN = get_basis_derivatives(Hex27(), Lagrange{2}(), Vec{3}((0.2, 0.3, -0.1)))
        sum_grad = sum(dN)
        @test sum_grad[1] ≈ 0.0 atol = 1e-14
        @test sum_grad[2] ≈ 0.0 atol = 1e-14
        @test sum_grad[3] ≈ 0.0 atol = 1e-14
    end

    @testset "Gradient return types" begin
        # Test that gradients have correct Vec type
        dN = get_basis_derivatives(Tri3(), Lagrange{1}(), Vec{2}((0.5, 0.25)))
        @test dN isa SVector{3,Vec{2,Float64}}
        @test dN[1] isa Vec{2,Float64}

        dN = get_basis_derivatives(Hex8(), Lagrange{1}(), Vec{3}((0.0, 0.0, 0.0)))
        @test dN isa SVector{8,Vec{3,Float64}}
        @test dN[1] isa Vec{3,Float64}
    end

    @testset "Numerical gradient check (finite differences)" begin
        # Simple finite difference check for Quad4
        h = 1e-8
        xi = Vec{2}((0.3, 0.4))

        # Analytical gradient
        dN = get_basis_derivatives(Quad4(), Lagrange{1}(), xi)

        # Finite difference in u-direction
        N_plus = get_basis_functions(Quad4(), Lagrange{1}(), Vec{2}((xi[1] + h, xi[2])))
        N_minus = get_basis_functions(Quad4(), Lagrange{1}(), Vec{2}((xi[1] - h, xi[2])))
        dN_u_fd = (N_plus - N_minus) / (2h)

        for i in 1:4
            @test dN[i][1] ≈ dN_u_fd[i] atol = 1e-6
        end
    end
end
