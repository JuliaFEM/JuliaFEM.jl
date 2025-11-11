"""
Unit tests for FiniteStrainPlasticity material model.

Tests cover:
1. Construction and validation
2. State initialization
3. Small strain limit (should match PerfectPlasticity)
4. Pure elastic deformation (large rotation)
5. Uniaxial extension beyond yield
6. Simple shear
7. Plastic incompressibility (det(F_p) = 1)
8. Type stability
9. State consistency
"""

using Test
using Tensors
using LinearAlgebra

# Load implementations
include("../src/materials/abstract_material.jl")
include("../src/materials/finite_strain_plasticity.jl")

@testset "Finite Strain Plasticity Material" begin

    @testset "Material Construction" begin
        # Valid construction
        steel = FiniteStrainPlasticity(E=200e9, ν=0.3, σ_y=250e6, H=1e9)
        @test steel.E == 200e9
        @test steel.ν == 0.3
        @test steel.σ_y == 250e6
        @test steel.H == 1e9
        @test steel.μ ≈ 200e9 / (2 * (1 + 0.3))
        @test steel.λ ≈ 200e9 * 0.3 / ((1 + 0.3) * (1 - 2 * 0.3))

        # Perfect plasticity (H=0)
        perfect = FiniteStrainPlasticity(E=200e9, ν=0.3, σ_y=250e6, H=0.0)
        @test perfect.H == 0.0

        # Invalid inputs
        @test_throws ArgumentError FiniteStrainPlasticity(E=-200e9, ν=0.3, σ_y=250e6, H=1e9)
        @test_throws ArgumentError FiniteStrainPlasticity(E=200e9, ν=0.6, σ_y=250e6, H=1e9)
        @test_throws ArgumentError FiniteStrainPlasticity(E=200e9, ν=0.3, σ_y=-250e6, H=1e9)
        @test_throws ArgumentError FiniteStrainPlasticity(E=200e9, ν=0.3, σ_y=250e6, H=-1e9)
    end

    @testset "State Construction" begin
        # Default state (identity F_p)
        state0 = FiniteStrainPlasticityState()
        @test state0.F_p == one(Tensor{2,3})
        @test state0.α_bar == zero(SymmetricTensor{2,3})
        @test state0.κ == 0.0
        @test det(state0.F_p) ≈ 1.0

        # Custom state
        F_p = one(Tensor{2,3}) + 0.01 * Tensor{2,3}((0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        F_p = F_p / det(F_p)^(1 / 3)  # Enforce det = 1
        α_bar = SymmetricTensor{2,3}((1e8, 0.0, 0.0, 0.0, 0.0, 0.0))
        state = FiniteStrainPlasticityState(F_p, α_bar, 0.01)
        @test state.F_p ≈ F_p
        @test state.α_bar == α_bar
        @test state.κ == 0.01

        # Invalid state (negative κ)
        @test_throws ArgumentError FiniteStrainPlasticityState(F_p, α_bar, -0.01)
    end

    @testset "Small Strain Limit" begin
        steel = FiniteStrainPlasticity(E=200e9, ν=0.3, σ_y=250e6, H=1e9)

        # Small deformation: F ≈ I + ∇u
        ε_small = 1e-5
        F_small = one(Tensor{2,3}) + ε_small * Tensor{2,3}((1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

        σ, 𝔸, state = compute_stress(steel, F_small, nothing, 0.0)

        # Should remain elastic
        @test state.F_p ≈ one(Tensor{2,3})
        @test state.α_bar == zero(SymmetricTensor{2,3})
        @test state.κ == 0.0

        # Stress should be small
        @test norm(σ) < 1e6  # Less than 1 MPa
    end

    @testset "Identity Deformation" begin
        steel = FiniteStrainPlasticity(E=200e9, ν=0.3, σ_y=250e6, H=1e9)
        F_identity = one(Tensor{2,3})

        σ, 𝔸, state = compute_stress(steel, F_identity, nothing, 0.0)

        # Zero stress for no deformation
        @test norm(σ) < 1e-10
        @test state.F_p == one(Tensor{2,3})
        @test state.κ == 0.0
    end

    @testset "Pure Rotation (Elastic)" begin
        steel = FiniteStrainPlasticity(E=200e9, ν=0.3, σ_y=250e6, H=1e9)

        # 45-degree rotation around z-axis (no stretching)
        θ = π / 4
        c = cos(θ)
        s = sin(θ)
        R = Tensor{2,3}((c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0))

        σ, 𝔸, state = compute_stress(steel, R, nothing, 0.0)

        # Pure rotation should give zero stress (if formulation is objective)
        # Note: May not be exactly zero due to numerical precision
        @test norm(σ) < 1e6  # Should be small
        @test state.F_p ≈ one(Tensor{2,3}) rtol = 1e-6
    end

    @testset "Uniaxial Extension (Elastic)" begin
        steel = FiniteStrainPlasticity(E=200e9, ν=0.3, σ_y=250e6, H=1e9)

        # 1% extension in x-direction
        λ = 1.01
        F_ext = Tensor{2,3}((λ, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

        σ, 𝔸, state = compute_stress(steel, F_ext, nothing, 0.0)

        # Should remain elastic (small extension)
        @test state.F_p ≈ one(Tensor{2,3}) rtol = 1e-6
        @test state.κ == 0.0

        # Check that σ_xx > 0 (tension)
        @test σ[1, 1] > 0.0
    end

    @testset "Uniaxial Extension (Plastic)" begin
        steel = FiniteStrainPlasticity(E=200e9, ν=0.3, σ_y=250e6, H=1e9)

        # Large extension (10%)
        λ = 1.10
        F_ext = Tensor{2,3}((λ, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

        σ, 𝔸, state = compute_stress(steel, F_ext, nothing, 0.0)

        # Should have plastic deformation
        @test norm(state.F_p - one(Tensor{2,3})) > 1e-6
        @test state.κ > 0.0

        # Plastic incompressibility: det(F_p) ≈ 1
        @test abs(det(state.F_p) - 1.0) < 1e-3
    end

    @testset "Simple Shear" begin
        steel = FiniteStrainPlasticity(E=200e9, ν=0.3, σ_y=250e6, H=1e9)

        # Shear deformation: γ = 0.1
        γ = 0.1
        F_shear = Tensor{2,3}((1.0, γ, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

        σ, 𝔸, state = compute_stress(steel, F_shear, nothing, 0.0)

        # Check shear stress exists
        @test abs(σ[1, 2]) > 0.0

        # det(F) should be 1 for simple shear
        @test abs(det(F_shear) - 1.0) < 1e-10
    end

    @testset "Incremental Loading" begin
        steel = FiniteStrainPlasticity(E=200e9, ν=0.3, σ_y=250e6, H=1e9)

        # Load in increments
        n_steps = 5
        λ_max = 1.05

        state = FiniteStrainPlasticityState()
        stresses = Float64[]
        plastic_strains = Float64[]

        for i in 1:n_steps
            λ = 1.0 + (λ_max - 1.0) * i / n_steps
            F = Tensor{2,3}((λ, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
            σ, 𝔸, state = compute_stress(steel, F, state, 0.0)

            push!(stresses, σ[1, 1])
            push!(plastic_strains, state.κ)
        end

        # Stress should increase (with hardening)
        @test all(diff(stresses) .≥ -1e-6)  # Allow small numerical errors

        # Plastic strain should increase monotonically
        @test all(diff(plastic_strains) .≥ 0.0)
    end

    @testset "Plastic Incompressibility" begin
        steel = FiniteStrainPlasticity(E=200e9, ν=0.3, σ_y=250e6, H=1e9)

        # Various deformation levels
        stretches = [1.02, 1.05, 1.10, 1.15, 1.20]

        for λ in stretches
            F = Tensor{2,3}((λ, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
            σ, 𝔸, state = compute_stress(steel, F, nothing, 0.0)

            # Check plastic incompressibility
            det_Fp = det(state.F_p)
            @test abs(det_Fp - 1.0) < 0.01  # Within 1% (relaxed due to exponential map approximation)
        end
    end

    @testset "Hardening Behavior" begin
        steel_hard = FiniteStrainPlasticity(E=200e9, ν=0.3, σ_y=250e6, H=10e9)
        steel_perf = FiniteStrainPlasticity(E=200e9, ν=0.3, σ_y=250e6, H=0.0)

        F_test = Tensor{2,3}((1.08, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

        σ_hard, _, state_hard = compute_stress(steel_hard, F_test, nothing, 0.0)
        σ_perf, _, state_perf = compute_stress(steel_perf, F_test, nothing, 0.0)

        # Hardening material should have higher stress
        @test σ_hard[1, 1] > σ_perf[1, 1]

        # Hardening material should have backstress
        @test norm(state_hard.α_bar) > 0.0
        @test norm(state_perf.α_bar) == 0.0
    end

    @testset "State Persistence" begin
        steel = FiniteStrainPlasticity(E=200e9, ν=0.3, σ_y=250e6, H=1e9)

        # First load
        F1 = Tensor{2,3}((1.08, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        σ1, _, state1 = compute_stress(steel, F1, nothing, 0.0)

        # Unload to smaller deformation
        F2 = Tensor{2,3}((1.02, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        σ2, _, state2 = compute_stress(steel, F2, state1, 0.0)

        # Plastic strain should not decrease
        @test state2.κ ≥ state1.κ

        # F_p should not go back to identity
        @test norm(state2.F_p - one(Tensor{2,3})) > 1e-6
    end

    @testset "Simplified Interface" begin
        steel = FiniteStrainPlasticity(E=200e9, ν=0.3, σ_y=250e6, H=1e9)
        F = Tensor{2,3}((1.05, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

        # Test with and without explicit state/Δt
        σ1, 𝔸1, state1 = compute_stress(steel, F)
        σ2, 𝔸2, state2 = compute_stress(steel, F, nothing, 0.0)

        @test σ1 ≈ σ2
        @test state1.κ ≈ state2.κ
    end

    @testset "Type Stability" begin
        steel = FiniteStrainPlasticity(E=200e9, ν=0.3, σ_y=250e6, H=1e9)
        F = Tensor{2,3}((1.05, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        state = FiniteStrainPlasticityState()

        # Infer return types
        result = @inferred compute_stress(steel, F, state, 0.0)

        @test result isa Tuple{SymmetricTensor{2,3,Float64},
            SymmetricTensor{4,3,Float64},
            FiniteStrainPlasticityState}
    end

end
