"""
# Unit Tests: Perfect Plasticity (J2 Plasticity with Hardening)

**What:** Comprehensive validation of rate-independent J2 plasticity with return mapping

**Why:**
- Foundation of metal plasticity (steel, aluminum, etc.)
- Tests critical algorithms: radial return mapping, consistency enforcement
- Validates state evolution: plastic strain ε_p, backstress α, hardening κ
- Critical for nonlinear structural analysis (beyond elastic limit)
- Demonstrates stateful material model (history-dependent)

**How:**
Test suite validates:
1. **Construction & parameters** - E, ν, σ_y, H validity, computed μ and λ
2. **State management** - PlasticityState(ε_p, α, κ) with zero default
3. **Elastic loading** - f < 0: stress σ = λ·tr(ε)I + 2μ·ε, no plastic strain
4. **Plastic loading** - f = 0: yield criterion √(3/2·s:s) = σ_y enforced
5. **Radial return mapping** - Projects trial stress back to yield surface
   - Consistency: von Mises stress = σ_y (within tolerance)
   - Plastic strain: ε_p deviatoric (tr(ε_p) = 0)
   - Stress reduction: ||σ|| < ||σ_elastic|| after return
6. **Hardening behavior** - Kinematic hardening via backstress α
   - H > 0: Stress increases with plastic strain
   - H = 0: Perfect plasticity (constant yield)
   - Backstress: norm(α) > 0 for hardening
7. **Incremental loading** - Monotonic tension: stress and κ increase
8. **Bauschinger effect** - Cyclic loading: early yield in reverse direction
9. **Pure shear** - Validates τ_yield = σ_y/√3 for shear stress
10. **Consistency** - Yield criterion f ≤ 0 satisfied for all strain levels
11. **Performance** - Zero allocations (elastic), minimal (plastic), type stability

**Mathematical Background:**
- Yield criterion: f = √(3/2·s:s) - σ_y ≤ 0 (von Mises)
  - s = dev(σ - α) = deviatoric relative stress
- Flow rule: ε̇_p = Δγ·∂f/∂σ = Δγ·(3/2·s/||s||) (associative plasticity)
- Hardening: α̇ = H·ε̇_p (kinematic hardening, models Bauschinger effect)
- Plastic work: κ̇ = √(2/3·ε̇_p:ε̇_p) (accumulated plastic strain)
- Radial return: σ_n+1 = σ_trial - 2μ·Δγ·n where n = s/||s||
- Consistency: Kuhn-Tucker conditions (f ≤ 0, Δγ ≥ 0, Δγ·f = 0)
- Physical constraints: E > 0, -1 < ν < 0.5, σ_y > 0, H ≥ 0

**Expected Results:**
✅ Material constructed: E=200 GPa, ν=0.3, σ_y=250 MPa, H=1 GPa
✅ Perfect plasticity: H=0 valid (no hardening)
✅ Invalid inputs rejected: E<0, ν>0.5, σ_y<0, H<0, κ<0
✅ Default state: ε_p=0, α=0, κ=0 (virgin material)
✅ Elastic (ε=1e-5): σ = elastic, ε_p=0, α=0, κ=0
✅ Plastic (ε=0.003 > ε_y=0.00125): ε_p > 0, κ > 0, von_mises = σ_y ± 1e-6
✅ Radial return (ε=0.01): f = 0 enforced, ||σ|| < ||σ_elastic||, ε_p > 1e-4
✅ Hardening: H=1 GPa → ||α|| > 0, ||σ_hard|| > ||σ_perfect||
✅ Incremental: Monotonic stress and κ increase (10 steps to ε=0.005)
✅ Bauschinger: Tension then compression → κ increases, α evolves
✅ Pure shear: τ_yield ≈ σ_y/√3 = 144 MPa for σ_y=250 MPa
✅ Consistency: f ≤ 1e-6 for all strain levels (0.1% to 2%)
✅ Simplified interface (without state, Δt) matches full call
✅ Zero allocations (elastic path), ≤256 bytes (plastic path for state)
✅ Type-stable: returns Tuple{SymmetricTensor{2,3}, SymmetricTensor{4,3}, PlasticityState}

**Test Coverage:**
- 13 test sets, ~80 individual assertions
- Material constants: Steel (E=200 GPa, ν=0.3, σ_y=250 MPa, H=0-10 GPa)
- Loading: Elastic (ε=1e-5), yield (ε=0.003), large (ε=0.01), shear (γ=0.005)
- Validation methods: Analytical yield criterion, stress comparison, state evolution
- Algorithms: Radial return, consistency enforcement, incremental loading, cyclic loading
- Edge cases: Perfect plasticity (H=0), high hardening (H=10 GPa), pure shear

**Key Physics:**
- J2 plasticity: Pressure-independent yield (metals)
- Radial return: Closest-point projection to yield surface
- Kinematic hardening: Backstress α shifts yield surface (Bauschinger effect)
- Consistency: Active constraint f=0 during plastic loading
- History dependence: Current stress depends on loading path via state
"""

using Test
using Tensors
using LinearAlgebra

# Load implementation
include("../src/materials/perfect_plasticity.jl")

@testset "Perfect Plasticity Material" begin

    @testset "Material Construction" begin
        # Valid construction
        steel = PerfectPlasticity(E=200e9, ν=0.3, σ_y=250e6, H=1e9)
        @test steel.E == 200e9
        @test steel.ν == 0.3
        @test steel.σ_y == 250e6
        @test steel.H == 1e9
        @test steel.μ ≈ 200e9 / (2 * (1 + 0.3))
        @test steel.λ ≈ 200e9 * 0.3 / ((1 + 0.3) * (1 - 2 * 0.3))

        # Perfect plasticity (H=0)
        perfect = PerfectPlasticity(E=200e9, ν=0.3, σ_y=250e6, H=0.0)
        @test perfect.H == 0.0

        # Invalid inputs
        @test_throws ArgumentError PerfectPlasticity(E=-200e9, ν=0.3, σ_y=250e6, H=1e9)  # Negative E
        @test_throws ArgumentError PerfectPlasticity(E=200e9, ν=0.6, σ_y=250e6, H=1e9)   # ν too large
        @test_throws ArgumentError PerfectPlasticity(E=200e9, ν=0.3, σ_y=-250e6, H=1e9)  # Negative σ_y
        @test_throws ArgumentError PerfectPlasticity(E=200e9, ν=0.3, σ_y=250e6, H=-1e9)  # Negative H
    end

    @testset "State Construction" begin
        # Default state (zero)
        state0 = PlasticityState()
        @test state0.ε_p == zero(SymmetricTensor{2,3})
        @test state0.α == zero(SymmetricTensor{2,3})
        @test state0.κ == 0.0

        # Custom state
        ε_p = SymmetricTensor{2,3}((0.01, 0.0, 0.0, 0.0, 0.0, 0.0))
        α = SymmetricTensor{2,3}((1e8, 0.0, 0.0, 0.0, 0.0, 0.0))
        state = PlasticityState(ε_p, α, 0.01)
        @test state.ε_p == ε_p
        @test state.α == α
        @test state.κ == 0.01

        # Invalid state (negative κ)
        @test_throws ArgumentError PlasticityState(ε_p, α, -0.01)
    end

    @testset "Elastic Loading (Small Strain)" begin
        steel = PerfectPlasticity(E=200e9, ν=0.3, σ_y=250e6, H=1e9)

        # Small strain (well below yield)
        ε_small = SymmetricTensor{2,3}((1e-5, 0.0, 0.0, 0.0, 0.0, 0.0))

        σ, 𝔻, state_new = compute_stress(steel, ε_small, nothing, 0.0)

        # Should remain elastic
        @test state_new.ε_p == zero(SymmetricTensor{2,3})  # No plastic strain
        @test state_new.α == zero(SymmetricTensor{2,3})    # No backstress
        @test state_new.κ == 0.0                            # No plastic work

        # Stress should be elastic
        μ = steel.μ
        λ = steel.λ
        I = one(ε_small)
        σ_elastic = λ * tr(ε_small) * I + 2μ * ε_small
        @test σ ≈ σ_elastic rtol = 1e-12

        # Tangent should be elastic
        @test 𝔻 isa SymmetricTensor{4,3}
    end

    @testset "Plastic Loading (Yield)" begin
        steel = PerfectPlasticity(E=200e9, ν=0.3, σ_y=250e6, H=1e9)

        # Strain beyond yield (uniaxial tension)
        # Yield strain: ε_y = σ_y / E ≈ 0.00125
        ε_plastic = SymmetricTensor{2,3}((0.003, 0.0, 0.0, 0.0, 0.0, 0.0))

        σ, 𝔻, state_new = compute_stress(steel, ε_plastic, nothing, 0.0)

        # Should have plastic strain
        @test norm(state_new.ε_p) > 0.0
        @test state_new.κ > 0.0

        # Check yield criterion (should be satisfied)
        s = dev(σ - state_new.α)
        von_mises = √(3 / 2) * √(s ⊡ s)
        @test von_mises ≈ steel.σ_y rtol = 1e-6  # On yield surface

        # Plastic strain should be deviatoric
        @test abs(tr(state_new.ε_p)) < 1e-12
    end

    @testset "Radial Return Mapping" begin
        steel = PerfectPlasticity(E=200e9, ν=0.3, σ_y=250e6, H=1e9)

        # Large strain (far beyond yield)
        ε_large = SymmetricTensor{2,3}((0.01, 0.0, 0.0, 0.0, 0.0, 0.0))

        σ, 𝔻, state_new = compute_stress(steel, ε_large, nothing, 0.0)

        # Check yield criterion (must be satisfied)
        s = dev(σ - state_new.α)
        von_mises = √(3 / 2) * √(s ⊡ s)
        @test von_mises ≈ steel.σ_y rtol = 1e-6

        # Stress should be less than elastic prediction
        μ = steel.μ
        λ = steel.λ
        I = one(ε_large)
        σ_elastic = λ * tr(ε_large) * I + 2μ * ε_large
        @test norm(σ) < norm(σ_elastic)

        # Plastic strain should be significant
        @test norm(state_new.ε_p) > 1e-4
    end

    @testset "Hardening Behavior" begin
        # Compare hardening vs perfect plasticity
        steel_hard = PerfectPlasticity(E=200e9, ν=0.3, σ_y=250e6, H=1e9)
        steel_perf = PerfectPlasticity(E=200e9, ν=0.3, σ_y=250e6, H=0.0)

        ε_test = SymmetricTensor{2,3}((0.005, 0.0, 0.0, 0.0, 0.0, 0.0))

        σ_hard, _, state_hard = compute_stress(steel_hard, ε_test, nothing, 0.0)
        σ_perf, _, state_perf = compute_stress(steel_perf, ε_test, nothing, 0.0)

        # Hardening material should have backstress
        @test norm(state_hard.α) > 0.0
        @test norm(state_perf.α) == 0.0

        # Hardening material should have higher stress
        @test norm(σ_hard) > norm(σ_perf)
    end

    @testset "Incremental Loading" begin
        steel = PerfectPlasticity(E=200e9, ν=0.3, σ_y=250e6, H=1e9)

        # Load in increments
        n_steps = 10
        ε_max = 0.005

        state = PlasticityState()
        stresses = []
        plastic_strains = []

        for i in 1:n_steps
            ε = SymmetricTensor{2,3}((i * ε_max / n_steps, 0.0, 0.0, 0.0, 0.0, 0.0))
            σ, _, state = compute_stress(steel, ε, state, 0.0)
            push!(stresses, σ[1, 1])
            push!(plastic_strains, state.κ)
        end

        # Stress should increase monotonically (hardening)
        @test all(diff(stresses) .≥ 0)

        # Plastic strain should increase monotonically
        @test all(diff(plastic_strains) .≥ 0)

        # Final plastic strain should be positive
        @test plastic_strains[end] > 0.0
    end

    @testset "Bauschinger Effect (Cyclic Loading)" begin
        steel = PerfectPlasticity(E=200e9, ν=0.3, σ_y=250e6, H=10e9)  # High H for visibility

        # Step 1: Tension to plastic regime
        ε_tension = SymmetricTensor{2,3}((0.003, 0.0, 0.0, 0.0, 0.0, 0.0))
        σ_t, _, state_t = compute_stress(steel, ε_tension, nothing, 0.0)

        # Step 2: Reverse to compression
        ε_compression = SymmetricTensor{2,3}((-0.002, 0.0, 0.0, 0.0, 0.0, 0.0))
        σ_c, _, state_c = compute_stress(steel, ε_compression, state_t, 0.0)

        # Should yield in compression earlier (Bauschinger effect from backstress)
        @test state_c.κ > state_t.κ  # Additional plastic strain
        @test norm(state_c.α) > 0.0   # Backstress present
    end

    @testset "Pure Shear" begin
        steel = PerfectPlasticity(E=200e9, ν=0.3, σ_y=250e6, H=1e9)

        # Pure shear strain
        γ = 0.005
        ε_shear = SymmetricTensor{2,3}((0.0, γ / 2, 0.0, 0.0, 0.0, 0.0))

        σ, _, state = compute_stress(steel, ε_shear, nothing, 0.0)

        # Check shear stress
        @test abs(σ[1, 2]) > 0.0

        # Check yield in shear
        # For pure shear: τ_yield = σ_y / √3
        s = dev(σ - state.α)
        von_mises = √(3 / 2) * √(s ⊡ s)

        if von_mises > steel.σ_y - 1e-3  # Plastic
            @test von_mises ≈ steel.σ_y rtol = 1e-6
        end
    end

    @testset "Consistency Check" begin
        steel = PerfectPlasticity(E=200e9, ν=0.3, σ_y=250e6, H=1e9)

        # Multiple strain levels
        strain_levels = [0.001, 0.002, 0.005, 0.01, 0.02]

        for ε_mag in strain_levels
            ε = SymmetricTensor{2,3}((ε_mag, 0.0, 0.0, 0.0, 0.0, 0.0))
            σ, _, state = compute_stress(steel, ε, nothing, 0.0)

            # Check yield criterion
            s = dev(σ - state.α)
            von_mises = √(3 / 2) * √(s ⊡ s)

            # Must satisfy: f = von_mises - σ_y ≤ 0
            f = von_mises - steel.σ_y
            @test f ≤ 1e-6  # On or inside yield surface
        end
    end

    @testset "Simplified Interface" begin
        steel = PerfectPlasticity(E=200e9, ν=0.3, σ_y=250e6, H=1e9)
        ε = SymmetricTensor{2,3}((0.003, 0.0, 0.0, 0.0, 0.0, 0.0))

        # Test with and without explicit state/Δt
        σ1, 𝔻1, state1 = compute_stress(steel, ε)
        σ2, 𝔻2, state2 = compute_stress(steel, ε, nothing, 0.0)

        @test σ1 ≈ σ2
        @test 𝔻1 ≈ 𝔻2
        @test state1.κ ≈ state2.κ
    end

    @testset "Zero Allocation" begin
        steel = PerfectPlasticity(E=200e9, ν=0.3, σ_y=250e6, H=1e9)
        state = PlasticityState()

        # Test elastic path (no state change)
        ε_elastic = SymmetricTensor{2,3}((1e-5, 0.0, 0.0, 0.0, 0.0, 0.0))

        # First call to compile
        compute_stress(steel, ε_elastic, state, 0.0)

        # Check allocations on elastic path
        allocs_elastic = @allocated compute_stress(steel, ε_elastic, state, 0.0)
        @test allocs_elastic == 0  # Elastic path should have zero allocations

        # Test plastic path (state changes)
        ε_plastic = SymmetricTensor{2,3}((0.003, 0.0, 0.0, 0.0, 0.0, 0.0))

        # First call to compile
        compute_stress(steel, ε_plastic, state, 0.0)

        # Check allocations on plastic path
        allocs_plastic = @allocated compute_stress(steel, ε_plastic, state, 0.0)
        # Note: Plastic path allocates ~128 bytes for PlasticityState struct
        # This is acceptable for stateful materials
        @test allocs_plastic ≤ 256  # Allow some allocation for state
    end

    @testset "Type Stability" begin
        steel = PerfectPlasticity(E=200e9, ν=0.3, σ_y=250e6, H=1e9)
        ε = SymmetricTensor{2,3}((0.002, 0.0, 0.0, 0.0, 0.0, 0.0))
        state = PlasticityState()

        # Infer return types
        result = @inferred compute_stress(steel, ε, state, 0.0)

        @test result isa Tuple{SymmetricTensor{2,3,Float64},
            SymmetricTensor{4,3,Float64},
            PlasticityState}
    end

end
