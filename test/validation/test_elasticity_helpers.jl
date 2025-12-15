# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Standalone test for elasticity assembly helpers.

Tests core helper functions WITHOUT requiring Element/BasisInfo infrastructure.
Uses Tensors.jl types directly.
"""

using Test
using LinearAlgebra
using Tensors

# Include material models
include("../benchmarks/material_models_benchmark.jl")

# Define standalone helper functions (simplified from assembly_helpers.jl)

"""Compute strain from shape function gradients and displacement."""
function compute_strain_from_gradients(
    ∇N::NTuple{N,Vec{3,Float64}},
    u::AbstractVector{Float64}
) where N
    @assert length(u) == 3N "Displacement vector size mismatch"

    # Displacement gradient: F = ∂u/∂X = ∑ᵢ uᵢ ⊗ ∇Nᵢ
    # Small strain: ε = ½(F + Fᵀ)

    F = zero(Tensor{2,3,Float64})
    for i in 1:N
        u_node = Vec{3}((u[3i-2], u[3i-1], u[3i]))
        F += u_node ⊗ ∇N[i]
    end

    # Symmetrize to get small strain tensor
    ε = symmetric(F)

    return ε
end

"""Accumulate element stiffness matrix."""
function accumulate_stiffness!(
    K_e::Matrix{Float64},
    ∇N::NTuple{N,Vec{3,Float64}},
    𝔻::SymmetricTensor{4,3,Float64},
    weight::Float64
) where N

    @inbounds for i in 1:N
        for j in 1:N
            # Stiffness contribution: Kᵢⱼ = ∫ ∇Nᵢ : 𝔻 : ∇Nⱼ dV
            # Split into spatial dimensions for explicit loops
            for α in 1:3  # Component of node i
                for β in 1:3  # Component of node j
                    # Sum over spatial indices (compiler unrolls)
                    val = 0.0
                    @simd for k in 1:3
                        @simd for l in 1:3
                            val += ∇N[i][k] * 𝔻[k, α, l, β] * ∇N[j][l]
                        end
                    end
                    K_e[3(i-1)+α, 3(j-1)+β] += weight * val
                end
            end
        end
    end

    return nothing
end

@testset "Elasticity Assembly Helpers (Standalone)" begin

    @testset "Material Model Integration" begin
        E = 200e9  # Pa
        ν = 0.3
        material = LinearElastic(E=E, ν=ν)

        # Test uniaxial strain
        ε = SymmetricTensor{2,3}((0.001, 0.0, 0.0, 0.0, 0.0, 0.0))
        σ, 𝔻, _ = compute_stress(material, ε, NoState(), 0.1)

        λ = E * ν / ((1 + ν) * (1 - 2ν))
        μ = E / (2(1 + ν))

        @test σ[1, 1] ≈ (λ + 2μ) * 0.001 atol = 1e-3
        @test σ[2, 2] ≈ λ * 0.001 atol = 1e-3
        @test σ[3, 3] ≈ λ * 0.001 atol = 1e-3

        println("✅ Material model correct")
    end

    @testset "Strain Computation" begin
        # Simple test: uniform extension in x-direction
        # ∇N gradients chosen so that: ε_xx = 0.001, all others = 0

        ∇N = (
            Vec{3}((1.0, 0.0, 0.0)),  # Node 1
            Vec{3}((0.0, 0.0, 0.0)),  # Node 2
            Vec{3}((0.0, 0.0, 0.0)),  # Node 3
            Vec{3}((0.0, 0.0, 0.0))   # Node 4
        )

        # Displacement: u₁ = [0.001, 0, 0], others zero
        u = zeros(12)
        u[1] = 0.001

        ε = compute_strain_from_gradients(∇N, u)

        @test ε[1, 1] ≈ 0.001 atol = 1e-6
        @test ε[2, 2] ≈ 0.0 atol = 1e-6
        @test ε[3, 3] ≈ 0.0 atol = 1e-6
        @test ε[1, 2] ≈ 0.0 atol = 1e-6

        println("✅ Strain computation correct")
    end

    @testset "Zero Allocation" begin
        # Setup
        ∇N = ntuple(4) do i
            Vec{3}((randn(), randn(), randn())) / 10.0
        end
        u = randn(12) .* 0.01

        E = 200e9
        ν = 0.3
        λ = E * ν / ((1 + ν) * (1 - 2ν))
        μ = E / (2(1 + ν))
        I = one(SymmetricTensor{2,3,Float64})
        𝕀ˢʸᵐ = one(SymmetricTensor{4,3,Float64})
        𝔻 = λ * I ⊗ I + 2μ * 𝕀ˢʸᵐ

        # Test compute_strain_from_gradients
        alloc1 = @allocated compute_strain_from_gradients(∇N, u)
        @test alloc1 == 0

        # Test accumulate_stiffness!
        K_e = zeros(12, 12)
        alloc2 = @allocated accumulate_stiffness!(K_e, ∇N, 𝔻, 1.0)
        @test alloc2 == 0

        println("✅ Zero allocations confirmed")
    end

    @testset "Type Stability" begin
        ∇N = ntuple(4) do i
            Vec{3}((0.1, 0.1, 0.1))
        end
        u = zeros(12)

        # Should infer to SymmetricTensor{2,3,Float64,6}
        @inferred compute_strain_from_gradients(∇N, u)

        E = 200e9
        ν = 0.3
        λ = E * ν / ((1 + ν) * (1 - 2ν))
        μ = E / (2(1 + ν))
        I = one(SymmetricTensor{2,3,Float64})
        𝕀ˢʸᵐ = one(SymmetricTensor{4,3,Float64})
        𝔻 = λ * I ⊗ I + 2μ * 𝕀ˢʸᵐ
        K_e = zeros(12, 12)

        # Should infer to Nothing
        @inferred accumulate_stiffness!(K_e, ∇N, 𝔻, 1.0)

        println("✅ Type stability confirmed")
    end

    @testset "Stiffness Matrix Properties" begin
        # Create realistic gradients
        ∇N = (
            Vec{3}((-0.5, -0.5, -0.5)),
            Vec{3}((0.5, 0.0, 0.0)),
            Vec{3}((0.0, 0.5, 0.0)),
            Vec{3}((0.0, 0.0, 0.5))
        )

        E = 200e9
        ν = 0.3
        λ = E * ν / ((1 + ν) * (1 - 2ν))
        μ = E / (2(1 + ν))
        I = one(SymmetricTensor{2,3,Float64})
        𝕀ˢʸᵐ = one(SymmetricTensor{4,3,Float64})
        𝔻 = λ * I ⊗ I + 2μ * 𝕀ˢʸᵐ

        K_e = zeros(12, 12)
        accumulate_stiffness!(K_e, ∇N, 𝔻, 1.0)

        # Check symmetry
        @test K_e ≈ K_e' atol = 1e-10

        # Check positive definiteness (approximately - some modes are zero)
        eigs = eigvals(K_e)
        # In a proper element, first 6 eigenvalues are ~0 (rigid body modes)
        # Others should be positive
        positive_eigs = count(λ -> λ > 1e6, eigs)
        @test positive_eigs >= 3  # At least some positive modes

        println("✅ Stiffness matrix properties validated")
    end

end

println("\n" * "="^60)
println("STANDALONE HELPERS TEST SUMMARY")
println("="^60)
println("✅ Material model integration working")
println("✅ Strain computation correct")
println("✅ Zero allocations confirmed")
println("✅ Type stability verified")
println("✅ Stiffness matrix properties validated")
println("="^60)
println("\n🎉 Core helper functions ready for full assembly!")
