# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using Test
using JuliaFEM
using Tensors

@testset "Strain Extraction" begin
    @testset "extract_strain" begin
        # Pure extension in x-direction
        ∇u = Tensor{2,3}((1.1, 0.0, 0.0,
                          0.0, 1.0, 0.0,
                          0.0, 0.0, 1.0))
        ε = extract_strain(∇u)

        @test ε isa SymmetricTensor{2,3}
        @test ε[1,1] ≈ 1.1  # ε_xx
        @test ε[2,2] ≈ 1.0  # ε_yy
        @test ε[3,3] ≈ 1.0  # ε_zz
        @test ε[1,2] ≈ 0.0  # ε_xy (symmetric)

        # Shear deformation
        ∇u_shear = Tensor{2,3}((1.0, 0.1, 0.0,
                                0.2, 1.0, 0.0,
                                0.0, 0.0, 1.0))
        ε_shear = extract_strain(∇u_shear)

        @test ε_shear[1,1] ≈ 1.0
        @test ε_shear[2,2] ≈ 1.0
        @test ε_shear[1,2] ≈ 0.15  # (0.1 + 0.2) / 2 = 0.15 (engineering shear strain / 2)
        @test ε_shear[2,1] ≈ 0.15  # Symmetric

        # General deformation
        ∇u_general = Tensor{2,3}((1.05, 0.03, 0.02,
                                  0.04, 1.08, 0.01,
                                  0.02, 0.03, 1.10))
        ε_general = extract_strain(∇u_general)

        @test ε_general[1,1] ≈ 1.05
        @test ε_general[2,2] ≈ 1.08
        @test ε_general[3,3] ≈ 1.10
        @test ε_general[1,2] ≈ (0.03 + 0.04) / 2
        @test ε_general[1,3] ≈ (0.02 + 0.02) / 2
        @test ε_general[2,3] ≈ (0.01 + 0.03) / 2
    end

    @testset "extract_strain_rate" begin
        # Constant strain rate (steady extension)
        ∇u̇ = Tensor{2,3}((0.01, 0.0, 0.0,
                          0.0, -0.005, 0.0,
                          0.0, 0.0, -0.005))
        ε̇ = extract_strain_rate(∇u̇)

        @test ε̇ isa SymmetricTensor{2,3}
        @test ε̇[1,1] ≈ 0.01     # Extension in x
        @test ε̇[2,2] ≈ -0.005   # Contraction in y (Poisson effect)
        @test ε̇[3,3] ≈ -0.005   # Contraction in z (Poisson effect)
        @test ε̇[1,2] ≈ 0.0

        # Shear rate
        ∇u̇_shear = Tensor{2,3}((0.0, 0.01, 0.0,
                                0.02, 0.0, 0.0,
                                0.0, 0.0, 0.0))
        ε̇_shear = extract_strain_rate(∇u̇_shear)

        @test ε̇_shear[1,2] ≈ (0.01 + 0.02) / 2
        @test ε̇_shear[2,1] ≈ (0.01 + 0.02) / 2  # Symmetric
    end

    @testset "Quasi-static strain rate from increments" begin
        # Simulate quasi-static loading with increments
        Δt = 1.0

        # Initial configuration
        ∇u_old = Tensor{2,3}((1.0, 0.0, 0.0,
                              0.0, 1.0, 0.0,
                              0.0, 0.0, 1.0))

        # New configuration after load step
        ∇u_new = Tensor{2,3}((1.01, 0.0, 0.0,
                              0.0, 0.99, 0.0,
                              0.0, 0.0, 0.99))

        # Compute strain rate from increment
        ∇u_rate = (∇u_new - ∇u_old) / Δt
        ε̇_quasi = extract_strain_rate(∇u_rate)

        @test ε̇_quasi[1,1] ≈ 0.01   # Strain rate from increment
        @test ε̇_quasi[2,2] ≈ -0.01  # Poisson effect
        @test ε̇_quasi[3,3] ≈ -0.01

        # This is needed for rate-dependent materials even in quasi-static!
        @test ε̇_quasi ≠ zero(SymmetricTensor{2,3})
    end

    @testset "Type stability" begin
        ∇u = Tensor{2,3}((1.1, 0.0, 0.0,
                          0.0, 1.0, 0.0,
                          0.0, 0.0, 1.0))

        # Type inference
        @inferred extract_strain(∇u)
        @inferred extract_strain_rate(∇u)

        # Zero allocations
        ε = extract_strain(∇u)  # Warmup
        allocs = @allocated extract_strain(∇u)
        @test allocs == 0

        ε̇ = extract_strain_rate(∇u)  # Warmup
        allocs = @allocated extract_strain_rate(∇u)
        @test allocs == 0
    end

    @testset "Compatibility with LocalField" begin
        # Test that strain extraction works with LocalField structure
        u = Vec{3}((0.1, 0.2, 0.3))
        ∇u = Tensor{2,3}((1.01, 0.02, 0.0,
                          0.03, 1.04, 0.0,
                          0.0, 0.0, 1.05))
        u̇ = zero(Vec{3})
        ∇u̇ = Tensor{2,3}((0.01, 0.0, 0.0,
                          0.0, 0.02, 0.0,
                          0.0, 0.0, 0.03))

        u_local = LocalField(u, ∇u, u̇, ∇u̇)

        # Extract strain from LocalField
        ε = extract_strain(u_local.gradient)
        ε̇ = extract_strain_rate(u_local.gradient_rate)

        @test ε isa SymmetricTensor{2,3}
        @test ε̇ isa SymmetricTensor{2,3}
        @test ε[1,1] ≈ 1.01
        @test ε̇[1,1] ≈ 0.01
    end
end
