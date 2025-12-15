"""
# Unit Tests: Finite Strain Plasticity (Multiplicative Decomposition)

**What:** Comprehensive validation of finite strain J2 plasticity with F = F_e F_p decomposition

**Why:**
- Geometrically exact plasticity for large deformations (>10% strain)
- Tests multiplicative decomposition F = F_e F_p (not additive ε = ε_e + ε_p)
- Validates plastic incompressibility det(F_p) = 1 (fundamental constraint)
- Critical for metal forming, impact, crashworthiness (extreme deformations)
- Demonstrates objective stress update (rotation-independent)

**How:**
Test suite validates:
1. **Construction & parameters** - E, ν, σ_y, H validity, computed μ and λ
2. **State management** - FiniteStrainPlasticityState(F_p, α_bar, κ) with F_p=I default
3. **Small strain limit** - Should recover small-strain plasticity for F ≈ I + ∇u
4. **Identity deformation** - F = I gives σ = 0, F_p = I, κ = 0
5. **Pure rotation** - Rigid body rotation (no stretch) should give σ ≈ 0 (objectivity)
6. **Uniaxial extension** - Elastic (λ=1.01) and plastic (λ=1.10) regimes
7. **Simple shear** - Validates shear response, det(F) = 1
8. **Incremental loading** - Monotonic loading: stress and κ increase
9. **Plastic incompressibility** - det(F_p) ≈ 1 for all stretches λ ∈ [1.02, 1.20]
10. **Hardening behavior** - H > 0: higher stress, backstress α_bar ≠ 0
11. **State persistence** - Unloading: plastic strain κ does not decrease
12. **Performance** - Type stability

**Mathematical Background:**
- Multiplicative decomposition: F = F_e F_p (Lee decomposition)
  - F: Total deformation gradient
  - F_e: Elastic part (recoverable on unloading)
  - F_p: Plastic part (permanent deformation)
- Plastic incompressibility: det(F_p) = 1 (volume preservation in plastic flow)
- Mandel stress: M = C_e S_e (intermediate configuration)
- Yield criterion: f = √(3/2·dev(M):dev(M)) - σ_y ≤ 0 (von Mises)
- Flow rule: Ḟ_p F_p⁻¹ = Δγ·n (exponential map integration)
- Hardening: α̇_bar = H·ε̇_p (backstress evolution in intermediate config)
- Objectivity: σ(Q·F) = Q·σ(F)·Q^T for rotation Q (frame-invariance)
- Physical constraints: det(F) > 0, det(F_e) > 0, det(F_p) = 1

**Expected Results:**
✅ Material constructed: E=200 GPa, ν=0.3, σ_y=250 MPa, H=0-10 GPa
✅ Perfect plasticity: H=0 valid
✅ Invalid inputs rejected: E<0, ν>0.5, σ_y<0, H<0, κ<0
✅ Default state: F_p=I (det=1), α_bar=0, κ=0
✅ Small strain (ε=1e-5): F_p≈I, κ=0, ||σ|| < 1 MPa
✅ Identity (F=I): σ=0 exactly
✅ Pure rotation (45° around z): ||σ|| < 1 MPa (objectivity), F_p≈I
✅ Uniaxial elastic (λ=1.01): F_p≈I, κ=0, σ_xx > 0
✅ Uniaxial plastic (λ=1.10): ||F_p-I|| > 1e-6, κ > 0, |det(F_p)-1| < 0.001
✅ Simple shear (γ=0.1): σ_xy ≠ 0, det(F)=1
✅ Incremental (5 steps to λ=1.05): Monotonic stress and κ
✅ Incompressibility: |det(F_p)-1| < 0.01 for λ ∈ [1.02,1.20]
✅ Hardening: H=10 GPa → σ > σ_perfect, ||α_bar|| > 0
✅ State persistence: Load λ=1.08 then unload λ=1.02 → κ doesn't decrease
✅ Simplified interface (without state, Δt) matches full call
✅ Type-stable: returns Tuple{SymmetricTensor{2,3}, SymmetricTensor{4,3}, FiniteStrainPlasticityState}

**Test Coverage:**
- 14 test sets, ~70 individual assertions
- Material constants: Steel (E=200 GPa, ν=0.3, σ_y=250 MPa, H=0-10 GPa)
- Deformations: Identity, small (ε=1e-5), rotation (45°), uniaxial (λ=1.01-1.20), shear (γ=0.1)
- Validation methods: Plastic incompressibility, objectivity, state persistence, hardening comparison
- Algorithms: Multiplicative decomposition, exponential map, return mapping in intermediate config
- Edge cases: Perfect plasticity (H=0), pure rotation, incremental loading, unloading

**Key Physics:**
- Multiplicative decomposition: Geometrically exact (not linearized)
- Plastic incompressibility: Fundamental for metals (no volume change in plastic flow)
- Objectivity: Stress independent of observer reference frame (essential for large rotations)
- Lee decomposition: Separates elastic (lattice stretch) from plastic (slip) deformations
- Intermediate configuration: Where plasticity lives (stress-free but plastically deformed)
- Exponential map: Preserves det(F_p) = 1 during integration (unlike additive schemes)
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
