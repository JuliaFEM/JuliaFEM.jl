# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using Test
using Tensors
using LinearAlgebra

# Use JuliaFEM for basis functions
using JuliaFEM

# Load our new deformation gradient code
include("../src/physics/deformation_gradient.jl")

@testset "Deformation Gradient - Low Level API" begin

    @testset "Identity case (u = 0)" begin
        # Unit cube element, no displacement
        X_nodes = (
            Vec(0.0, 0.0, 0.0),
            Vec(1.0, 0.0, 0.0),
            Vec(1.0, 1.0, 0.0),
            Vec(0.0, 1.0, 0.0),
            Vec(0.0, 0.0, 1.0),
            Vec(1.0, 0.0, 1.0),
            Vec(1.0, 1.0, 1.0),
            Vec(0.0, 1.0, 1.0)
        )

        # Zero displacement
        u_nodes = tuple([zero(Vec{3,Float64}) for _ in 1:8]...)

        # At element center ξ = (0, 0, 0)
        ξ = Vec(0.0, 0.0, 0.0)

        # Hex8 basis function derivatives at center (new API)
        dN_dξ = get_basis_derivatives(Hexahedron(), Lagrange{Hexahedron,1}(), ξ)

        # Compute Jacobian
        J = zero(Tensor{2,3,Float64,9})
        for i in 1:8
            J += X_nodes[i] ⊗ dN_dξ[i]
        end

        # Finite strain: Should give F = I + 0 = I
        F_finite = compute_deformation_gradient(X_nodes, u_nodes, dN_dξ, J, FiniteStrain())
        @test F_finite ≈ one(Tensor{2,3})
        @test det(F_finite) ≈ 1.0

        # Small strain: Should also give F = I
        F_small = compute_deformation_gradient(X_nodes, u_nodes, dN_dξ, J, SmallStrain())
        @test F_small ≈ one(Tensor{2,3})
        @test det(F_small) ≈ 1.0
    end

    @testset "Pure translation" begin
        # Unit cube
        X_nodes = (
            Vec(0.0, 0.0, 0.0),
            Vec(1.0, 0.0, 0.0),
            Vec(1.0, 1.0, 0.0),
            Vec(0.0, 1.0, 0.0),
            Vec(0.0, 0.0, 1.0),
            Vec(1.0, 0.0, 1.0),
            Vec(1.0, 1.0, 1.0),
            Vec(0.0, 1.0, 1.0)
        )

        # Uniform translation: u = (0.5, 0.5, 0.5) everywhere
        u_const = Vec(0.5, 0.5, 0.5)
        u_nodes = tuple([u_const for _ in 1:8]...)

        ξ = Vec(0.0, 0.0, 0.0)
        dN_dξ = get_basis_derivatives(Hexahedron(), Lagrange{Hexahedron,1}(), ξ)

        J = zero(Tensor{2,3,Float64,9})
        for i in 1:8
            J += X_nodes[i] ⊗ dN_dξ[i]
        end

        # Pure translation ⇒ ∇u = 0 ⇒ F = I
        F = compute_deformation_gradient(X_nodes, u_nodes, dN_dξ, J, FiniteStrain())
        @test F ≈ one(Tensor{2,3})
        @test det(F) ≈ 1.0
    end

    @testset "Pure stretch in x-direction" begin
        # Unit cube
        X_nodes = (
            Vec(0.0, 0.0, 0.0),
            Vec(1.0, 0.0, 0.0),
            Vec(1.0, 1.0, 0.0),
            Vec(0.0, 1.0, 0.0),
            Vec(0.0, 0.0, 1.0),
            Vec(1.0, 0.0, 1.0),
            Vec(1.0, 1.0, 1.0),
            Vec(0.0, 1.0, 1.0)
        )

        # Stretch: u_x = 0.1 * X (10% stretch in x)
        u_nodes = (
            Vec(0.0, 0.0, 0.0),  # u = 0.1 * 0 = 0
            Vec(0.1, 0.0, 0.0),  # u = 0.1 * 1 = 0.1
            Vec(0.1, 0.0, 0.0),  # u = 0.1 * 1 = 0.1
            Vec(0.0, 0.0, 0.0),  # u = 0.1 * 0 = 0
            Vec(0.0, 0.0, 0.0),  # u = 0.1 * 0 = 0
            Vec(0.1, 0.0, 0.0),  # u = 0.1 * 1 = 0.1
            Vec(0.1, 0.0, 0.0),  # u = 0.1 * 1 = 0.1
            Vec(0.0, 0.0, 0.0)   # u = 0.1 * 0 = 0
        )

        ξ = Vec(0.0, 0.0, 0.0)
        dN_dξ = get_basis_derivatives(Hexahedron(), Lagrange{Hexahedron,1}(), ξ)

        J = zero(Tensor{2,3,Float64,9})
        for i in 1:8
            J += X_nodes[i] ⊗ dN_dξ[i]
        end

        F = compute_deformation_gradient(X_nodes, u_nodes, dN_dξ, J, FiniteStrain())

        # Expected: F = [1.1  0  0]
        #               [0    1  0]
        #               [0    0  1]
        @test F[1, 1] ≈ 1.1 atol = 1e-10
        @test F[2, 2] ≈ 1.0 atol = 1e-10
        @test F[3, 3] ≈ 1.0 atol = 1e-10
        @test F[1, 2] ≈ 0.0 atol = 1e-10
        @test F[1, 3] ≈ 0.0 atol = 1e-10
        @test F[2, 3] ≈ 0.0 atol = 1e-10
        @test det(F) ≈ 1.1 atol = 1e-10
    end

    @testset "Simple shear" begin
        # Unit cube
        X_nodes = (
            Vec(0.0, 0.0, 0.0),
            Vec(1.0, 0.0, 0.0),
            Vec(1.0, 1.0, 0.0),
            Vec(0.0, 1.0, 0.0),
            Vec(0.0, 0.0, 1.0),
            Vec(1.0, 0.0, 1.0),
            Vec(1.0, 1.0, 1.0),
            Vec(0.0, 1.0, 1.0)
        )

        # Shear: u_x = 0.1 * y
        u_nodes = (
            Vec(0.0, 0.0, 0.0),   # y=0
            Vec(0.0, 0.0, 0.0),   # y=0
            Vec(0.1, 0.0, 0.0),   # y=1
            Vec(0.1, 0.0, 0.0),   # y=1
            Vec(0.0, 0.0, 0.0),   # y=0
            Vec(0.0, 0.0, 0.0),   # y=0
            Vec(0.1, 0.0, 0.0),   # y=1
            Vec(0.1, 0.0, 0.0)    # y=1
        )

        ξ = Vec(0.0, 0.0, 0.0)
        dN_dξ = get_basis_derivatives(Hexahedron(), Lagrange{Hexahedron,1}(), ξ)

        J = zero(Tensor{2,3,Float64,9})
        for i in 1:8
            J += X_nodes[i] ⊗ dN_dξ[i]
        end

        F = compute_deformation_gradient(X_nodes, u_nodes, dN_dξ, J, FiniteStrain())

        # Expected: F = [1    0.1  0]
        #               [0    1    0]
        #               [0    0    1]
        @test F[1, 1] ≈ 1.0 atol = 1e-10
        @test F[1, 2] ≈ 0.1 atol = 1e-10
        @test F[2, 2] ≈ 1.0 atol = 1e-10
        @test F[3, 3] ≈ 1.0 atol = 1e-10
        @test det(F) ≈ 1.0 atol = 1e-10
    end

    @testset "Small vs Finite strain difference" begin
        # Setup with significant displacement gradient
        X_nodes = (
            Vec(0.0, 0.0, 0.0),
            Vec(1.0, 0.0, 0.0),
            Vec(1.0, 1.0, 0.0),
            Vec(0.0, 1.0, 0.0),
            Vec(0.0, 0.0, 1.0),
            Vec(1.0, 0.0, 1.0),
            Vec(1.0, 1.0, 1.0),
            Vec(0.0, 1.0, 1.0)
        )

        # 20% stretch in x
        u_nodes = (
            Vec(0.0, 0.0, 0.0),
            Vec(0.2, 0.0, 0.0),
            Vec(0.2, 0.0, 0.0),
            Vec(0.0, 0.0, 0.0),
            Vec(0.0, 0.0, 0.0),
            Vec(0.2, 0.0, 0.0),
            Vec(0.2, 0.0, 0.0),
            Vec(0.0, 0.0, 0.0)
        )

        ξ = Vec(0.0, 0.0, 0.0)
        dN_dξ = get_basis_derivatives(Hexahedron(), Lagrange{Hexahedron,1}(), ξ)

        J = zero(Tensor{2,3,Float64,9})
        for i in 1:8
            J += X_nodes[i] ⊗ dN_dξ[i]
        end

        F_finite = compute_deformation_gradient(X_nodes, u_nodes, dN_dξ, J, FiniteStrain())
        F_small = compute_deformation_gradient(X_nodes, u_nodes, dN_dξ, J, SmallStrain())

        # Finite strain includes gradient
        @test F_finite[1, 1] ≈ 1.2 atol = 1e-10

        # Small strain ignores gradient
        @test F_small[1, 1] ≈ 1.0 atol = 1e-10

        # They should be different!
        @test !(F_finite ≈ F_small)
    end

    @testset "Physical constraint: det(F) > 0" begin
        # Physical deformation must preserve orientation
        X_nodes = (
            Vec(0.0, 0.0, 0.0),
            Vec(1.0, 0.0, 0.0),
            Vec(1.0, 1.0, 0.0),
            Vec(0.0, 1.0, 0.0),
            Vec(0.0, 0.0, 1.0),
            Vec(1.0, 0.0, 1.0),
            Vec(1.0, 1.0, 1.0),
            Vec(0.0, 1.0, 1.0)
        )

        # Small positive stretch
        u_nodes = (
            Vec(0.0, 0.0, 0.0),
            Vec(0.05, 0.0, 0.0),
            Vec(0.05, 0.0, 0.0),
            Vec(0.0, 0.0, 0.0),
            Vec(0.0, 0.0, 0.0),
            Vec(0.05, 0.0, 0.0),
            Vec(0.05, 0.0, 0.0),
            Vec(0.0, 0.0, 0.0)
        )

        ξ = Vec(0.0, 0.0, 0.0)
        dN_dξ = get_basis_derivatives(Hexahedron(), Lagrange{Hexahedron,1}(), ξ)

        J = zero(Tensor{2,3,Float64,9})
        for i in 1:8
            J += X_nodes[i] ⊗ dN_dξ[i]
        end

        F = compute_deformation_gradient(X_nodes, u_nodes, dN_dξ, J, FiniteStrain())

        @test det(F) > 0  # Physical requirement
    end
end

@testset "Deformation Gradient - Tet10 Element" begin

    @testset "Tet10: Identity case" begin
        # Regular tetrahedron nodes (4 corners + 6 edge midpoints)
        X_nodes = (
            Vec(0.0, 0.0, 0.0),              # 1: corner
            Vec(1.0, 0.0, 0.0),              # 2: corner
            Vec(0.0, 1.0, 0.0),              # 3: corner
            Vec(0.0, 0.0, 1.0),              # 4: corner
            Vec(0.5, 0.0, 0.0),              # 5: edge 1-2
            Vec(0.5, 0.5, 0.0),              # 6: edge 2-3
            Vec(0.0, 0.5, 0.0),              # 7: edge 3-1
            Vec(0.0, 0.0, 0.5),              # 8: edge 1-4
            Vec(0.5, 0.0, 0.5),              # 9: edge 2-4
            Vec(0.0, 0.5, 0.5)               # 10: edge 3-4
        )

        # Zero displacement
        u_nodes = tuple([zero(Vec{3,Float64}) for _ in 1:10]...)

        # At element centroid ξ = (1/4, 1/4, 1/4)
        ξ = Vec(0.25, 0.25, 0.25)

        # Tet10 basis function derivatives (new API)
        dN_dξ = get_basis_derivatives(Tetrahedron(), Lagrange{Tetrahedron,2}(), ξ)

        # Compute Jacobian
        J = zero(Tensor{2,3,Float64,9})
        for i in 1:10
            J += X_nodes[i] ⊗ dN_dξ[i]
        end

        F = compute_deformation_gradient(X_nodes, u_nodes, dN_dξ, J, FiniteStrain())

        @test F ≈ one(Tensor{2,3}) atol = 1e-10
        @test det(F) ≈ 1.0 atol = 1e-10
    end

    @testset "Tet10: Uniform stretch" begin
        # Regular tetrahedron
        X_nodes = (
            Vec(0.0, 0.0, 0.0),
            Vec(1.0, 0.0, 0.0),
            Vec(0.0, 1.0, 0.0),
            Vec(0.0, 0.0, 1.0),
            Vec(0.5, 0.0, 0.0),
            Vec(0.5, 0.5, 0.0),
            Vec(0.0, 0.5, 0.0),
            Vec(0.0, 0.0, 0.5),
            Vec(0.5, 0.0, 0.5),
            Vec(0.0, 0.5, 0.5)
        )

        # Isotropic expansion: u = 0.1 * X
        u_nodes = (
            Vec(0.0, 0.0, 0.0),
            Vec(0.1, 0.0, 0.0),
            Vec(0.0, 0.1, 0.0),
            Vec(0.0, 0.0, 0.1),
            Vec(0.05, 0.0, 0.0),
            Vec(0.05, 0.05, 0.0),
            Vec(0.0, 0.05, 0.0),
            Vec(0.0, 0.0, 0.05),
            Vec(0.05, 0.0, 0.05),
            Vec(0.0, 0.05, 0.05)
        )

        ξ = Vec(0.25, 0.25, 0.25)
        dN_dξ = get_basis_derivatives(Tetrahedron(), Lagrange{Tetrahedron,2}(), ξ)

        J = zero(Tensor{2,3,Float64,9})
        for i in 1:10
            J += X_nodes[i] ⊗ dN_dξ[i]
        end

        F = compute_deformation_gradient(X_nodes, u_nodes, dN_dξ, J, FiniteStrain())

        # Expected: F ≈ 1.1 * I
        @test F[1, 1] ≈ 1.1 atol = 1e-10
        @test F[2, 2] ≈ 1.1 atol = 1e-10
        @test F[3, 3] ≈ 1.1 atol = 1e-10
        @test abs(F[1, 2]) < 1e-10
        @test abs(F[1, 3]) < 1e-10
        @test abs(F[2, 3]) < 1e-10
        @test det(F) ≈ 1.1^3 atol = 1e-10
    end
end

@testset "Deformation Gradient - Zero Allocation" begin

    @testset "Verify zero allocations" begin
        # Setup
        X_nodes = (
            Vec(0.0, 0.0, 0.0),
            Vec(1.0, 0.0, 0.0),
            Vec(1.0, 1.0, 0.0),
            Vec(0.0, 1.0, 0.0),
            Vec(0.0, 0.0, 1.0),
            Vec(1.0, 0.0, 1.0),
            Vec(1.0, 1.0, 1.0),
            Vec(0.0, 1.0, 1.0)
        )

        u_nodes = (
            Vec(0.0, 0.0, 0.0),
            Vec(0.1, 0.0, 0.0),
            Vec(0.1, 0.0, 0.0),
            Vec(0.0, 0.0, 0.0),
            Vec(0.0, 0.0, 0.0),
            Vec(0.1, 0.0, 0.0),
            Vec(0.1, 0.0, 0.0),
            Vec(0.0, 0.0, 0.0)
        )

        ξ = Vec(0.0, 0.0, 0.0)
        dN_dξ = get_basis_derivatives(Hexahedron(), Lagrange{Hexahedron,1}(), ξ)

        J = zero(Tensor{2,3,Float64,9})
        for i in 1:8
            J += X_nodes[i] ⊗ dN_dξ[i]
        end

        # Warm up (compile)
        F = compute_deformation_gradient(X_nodes, u_nodes, dN_dξ, J, FiniteStrain())

        # Measure allocations
        allocs = @allocated compute_deformation_gradient(X_nodes, u_nodes, dN_dξ, J, FiniteStrain())

        @test allocs == 0  # Zero allocations!
    end
end
