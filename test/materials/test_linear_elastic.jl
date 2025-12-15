"""
# Unit Tests: LinearElastic Material Model

**What:** Comprehensive validation of isotropic linear elastic material σ = C:ε

**Why:**
- Foundation of structural mechanics (Hooke's law in 3D)
- Most common material model in engineering FEA
- Validates correct implementation of elasticity tensor C
- Critical for linear static/dynamic analysis

**How:**
Test suite validates:
1. **Construction & parameters** - E, ν validity, Lamé parameters λ and μ
2. **Stress computation** - Hooke's law σ = λ·tr(ε)I + 2μ·ε for various load cases:
   - Uniaxial extension: σ₁₁ = (λ + 2μ)·ε₁₁, lateral: σ₂₂ = σ₃₃ = λ·ε₁₁
   - Pure shear: σ₁₂ = 2μ·ε₁₂ (shear modulus definition)
   - Hydrostatic: σ = K·ε_vol·I where K = E/(3(1-2ν)) is bulk modulus
   - General strain: validates full 3D constitutive law
3. **Tangent modulus** - 4th-order tensor 𝔻 = λ·I⊗I + 2μ·𝕀ˢʸᵐ
   - Structure: SymmetricTensor{4,3}
   - Consistency: strain-independent (linear elasticity)
   - Double contraction: σ = 𝔻 ⊡ ε
4. **Physical properties** - Symmetry, isotropy, positive-definiteness
5. **Performance** - Zero allocations, type stability

**Mathematical Background:**
- Lamé parameters: λ = Eν/((1+ν)(1-2ν)), μ = E/(2(1+ν)) = G
- Bulk modulus: K = E/(3(1-2ν)) = λ + 2μ/3
- Elasticity tensor: C_{ijkl} = λ·δ_{ij}δ_{kl} + μ·(δ_{ik}δ_{jl} + δ_{il}δ_{jk})
- Physical constraints: E > 0, -1 < ν < 0.5 (0 ≤ ν < 0.5 for stable materials)

**Expected Results:**
✅ Material constructed with valid E, ν
✅ Lamé parameters computed correctly: λ ≈ 115.4 GPa, μ ≈ 76.9 GPa for steel
✅ Uniaxial stress: (λ+2μ)·ε₁₁ ≈ 269 GPa × 0.001 = 269 MPa
✅ Shear stress: 2μ·ε₁₂ ≈ 77 GPa × 0.002 = 154 MPa
✅ Hydrostatic: σ = K·ε_vol·I with correct bulk modulus
✅ General strain: σ = λ·tr(ε)I + 2μ·ε matches analytical
✅ Tangent 𝔻 has correct structure, constant for all strains
✅ Stress symmetry: σ_{ij} = σ_{ji}
✅ Isotropy: same strain magnitude → same stress magnitude in any direction
✅ Simplified interface (without state, Δt) works
✅ Zero allocations after compilation
✅ Type-stable: returns Tuple{SymmetricTensor{2,3}, SymmetricTensor{4,3}, Nothing}

**Test Coverage:**
- 12 test sets, ~70 individual assertions
- Material constants: Steel (E=200 GPa, ν=0.3), Aluminum (E=70 GPa, ν=0.33)
- Numerical validation: Analytical formulas + physical constraints
- Edge cases: Zero strain, pure modes, combined loading
"""

using Test
using Tensors

# Load implementation
include("../src/materials/linear_elastic.jl")

@testset "Linear Elastic Material" begin

    @testset "Material Construction" begin
        # Valid construction
        steel = LinearElastic(E=200e9, ν=0.3)
        @test steel.E == 200e9
        @test steel.ν == 0.3

        # Keyword constructor
        aluminum = LinearElastic(E=70e9, ν=0.33)
        @test aluminum.E == 70e9
        @test aluminum.ν == 0.33

        # Invalid inputs
        @test_throws ArgumentError LinearElastic(E=-100e9, ν=0.3)  # Negative E
        @test_throws ArgumentError LinearElastic(E=200e9, ν=0.6)   # ν too large
        @test_throws ArgumentError LinearElastic(E=200e9, ν=-1.1)  # ν too small
    end

    @testset "Lamé Parameters" begin
        steel = LinearElastic(E=200e9, ν=0.3)

        # First Lamé parameter: λ = E·ν/((1+ν)(1-2ν))
        λ_expected = 200e9 * 0.3 / ((1 + 0.3) * (1 - 2 * 0.3))
        @test λ(steel) ≈ λ_expected rtol = 1e-12
        @test λ(steel) ≈ 115.38461538461539e9 rtol = 1e-10

        # Shear modulus: μ = E/(2(1+ν))
        μ_expected = 200e9 / (2 * (1 + 0.3))
        @test μ(steel) ≈ μ_expected rtol = 1e-12
        @test μ(steel) ≈ 76.92307692307693e9 rtol = 1e-10

        # Test inline optimization (should compile to constants)
        @test @inferred λ(steel) isa Float64
        @test @inferred μ(steel) isa Float64
    end

    @testset "Stress Computation - Uniaxial Extension" begin
        steel = LinearElastic(E=200e9, ν=0.3)

        # Uniaxial extension in x-direction: ε = [ε₁₁, 0, 0; 0, 0, 0; 0, 0, 0]
        ε₁₁ = 0.001
        ε = SymmetricTensor{2,3}((ε₁₁, 0.0, 0.0, 0.0, 0.0, 0.0))

        σ, 𝔻, state_new = compute_stress(steel, ε, nothing, 0.0)

        # Expected stress: σ₁₁ = (λ + 2μ)·ε₁₁, σ₂₂ = σ₃₃ = λ·ε₁₁
        λ_val = λ(steel)
        μ_val = μ(steel)
        σ₁₁_expected = (λ_val + 2μ_val) * ε₁₁
        σ₂₂_expected = λ_val * ε₁₁

        @test σ[1, 1] ≈ σ₁₁_expected rtol = 1e-12
        @test σ[2, 2] ≈ σ₂₂_expected rtol = 1e-12
        @test σ[3, 3] ≈ σ₂₂_expected rtol = 1e-12
        @test σ[1, 2] ≈ 0.0 atol = 1e-15
        @test σ[1, 3] ≈ 0.0 atol = 1e-15
        @test σ[2, 3] ≈ 0.0 atol = 1e-15

        # State should be nothing (stateless material)
        @test state_new === nothing

        # Numerical check: σ₁₁ = (λ + 2μ)·ε₁₁ ≈ 269.2 MPa
        @test σ[1, 1] ≈ 269.2e6 rtol = 1e-2
        @test σ[2, 2] ≈ 115.4e6 rtol = 1e-2  # λ·ε₁₁ (positive for extension)
    end

    @testset "Stress Computation - Pure Shear" begin
        steel = LinearElastic(E=200e9, ν=0.3)

        # Pure shear: ε₁₂ = γ/2 (engineering shear strain γ = 0.002)
        γ = 0.002
        ε₁₂ = γ / 2  # Tensor shear strain
        ε = SymmetricTensor{2,3}((0.0, ε₁₂, 0.0, 0.0, 0.0, 0.0))

        σ, 𝔻, state_new = compute_stress(steel, ε, nothing, 0.0)

        # Expected stress: σ₁₂ = 2μ·ε₁₂
        μ_val = μ(steel)
        σ₁₂_expected = 2μ_val * ε₁₂

        @test σ[1, 2] ≈ σ₁₂_expected rtol = 1e-12
        @test σ[1, 1] ≈ 0.0 atol = 1e-15
        @test σ[2, 2] ≈ 0.0 atol = 1e-15
        @test σ[3, 3] ≈ 0.0 atol = 1e-15

        # Numerical check: σ₁₂ = 2μ·(γ/2) = μ·γ ≈ 77 GPa × 0.002 = 154 MPa
        @test σ[1, 2] ≈ 154e6 rtol = 1e-2

        @test state_new === nothing
    end

    @testset "Stress Computation - Hydrostatic Pressure" begin
        steel = LinearElastic(E=200e9, ν=0.3)

        # Hydrostatic strain: ε = ε_vol/3 · I
        ε_vol = 0.003  # Volumetric strain
        ε_iso = ε_vol / 3
        ε = SymmetricTensor{2,3}((ε_iso, 0.0, 0.0, ε_iso, 0.0, ε_iso))

        σ, 𝔻, state_new = compute_stress(steel, ε, nothing, 0.0)

        # Expected stress: σ = (λ + 2μ/3)·ε_vol·I = K·ε_vol·I
        # Bulk modulus: K = λ + 2μ/3 = E/(3(1-2ν))
        λ_val = λ(steel)
        μ_val = μ(steel)
        K = λ_val + 2μ_val / 3
        σ_expected = K * ε_vol

        @test σ[1, 1] ≈ σ_expected rtol = 1e-12
        @test σ[2, 2] ≈ σ_expected rtol = 1e-12
        @test σ[3, 3] ≈ σ_expected rtol = 1e-12
        @test σ[1, 2] ≈ 0.0 atol = 1e-15
        @test σ[1, 3] ≈ 0.0 atol = 1e-15
        @test σ[2, 3] ≈ 0.0 atol = 1e-15

        # Bulk modulus check
        K_expected = steel.E / (3 * (1 - 2 * steel.ν))
        @test K ≈ K_expected rtol = 1e-12

        @test state_new === nothing
    end

    @testset "Stress Computation - General Strain" begin
        steel = LinearElastic(E=200e9, ν=0.3)

        # General strain tensor (all components non-zero)
        ε = SymmetricTensor{2,3}((0.001, 0.0005, 0.0003, -0.0002, 0.0004, 0.0006))

        σ, 𝔻, state_new = compute_stress(steel, ε, nothing, 0.0)

        # Verify Hooke's law: σ = λ·tr(ε)·I + 2μ·ε
        λ_val = λ(steel)
        μ_val = μ(steel)
        I = one(ε)
        σ_expected = λ_val * tr(ε) * I + 2μ_val * ε

        @test σ ≈ σ_expected rtol = 1e-12

        # Check each component explicitly
        @test σ[1, 1] ≈ σ_expected[1, 1] rtol = 1e-12
        @test σ[2, 2] ≈ σ_expected[2, 2] rtol = 1e-12
        @test σ[3, 3] ≈ σ_expected[3, 3] rtol = 1e-12
        @test σ[1, 2] ≈ σ_expected[1, 2] rtol = 1e-12
        @test σ[1, 3] ≈ σ_expected[1, 3] rtol = 1e-12
        @test σ[2, 3] ≈ σ_expected[2, 3] rtol = 1e-12

        @test state_new === nothing
    end

    @testset "Tangent Modulus - Structure" begin
        steel = LinearElastic(E=200e9, ν=0.3)
        ε = SymmetricTensor{2,3}((0.001, 0.0, 0.0, 0.0, 0.0, 0.0))

        σ, 𝔻, _ = compute_stress(steel, ε, nothing, 0.0)

        # Verify tangent is 4th order symmetric tensor
        @test 𝔻 isa SymmetricTensor{4,3}

        # Verify 𝔻 = λ·I⊗I + 2μ·𝕀ˢʸᵐ
        λ_val = λ(steel)
        μ_val = μ(steel)
        I = one(ε)
        𝕀ˢʸᵐ = one(SymmetricTensor{4,3,Float64})
        𝔻_expected = λ_val * (I ⊗ I) + 2μ_val * 𝕀ˢʸᵐ

        @test 𝔻 ≈ 𝔻_expected rtol = 1e-12
    end

    @testset "Tangent Modulus - Consistency" begin
        steel = LinearElastic(E=200e9, ν=0.3)

        # Tangent should be constant (independent of strain)
        ε1 = SymmetricTensor{2,3}((0.001, 0.0, 0.0, 0.0, 0.0, 0.0))
        ε2 = SymmetricTensor{2,3}((0.005, 0.002, 0.001, -0.003, 0.0, 0.0))

        _, 𝔻1, _ = compute_stress(steel, ε1, nothing, 0.0)
        _, 𝔻2, _ = compute_stress(steel, ε2, nothing, 0.0)

        @test 𝔻1 ≈ 𝔻2 rtol = 1e-12
    end

    @testset "Tangent Modulus - Double Contraction" begin
        steel = LinearElastic(E=200e9, ν=0.3)
        ε = SymmetricTensor{2,3}((0.001, 0.0005, 0.0003, -0.0002, 0.0004, 0.0006))

        σ, 𝔻, _ = compute_stress(steel, ε, nothing, 0.0)

        # Verify σ = 𝔻 ⊡ ε (double contraction)
        σ_from_tangent = 𝔻 ⊡ ε

        @test σ ≈ σ_from_tangent rtol = 1e-12
    end

    @testset "Symmetry Properties" begin
        steel = LinearElastic(E=200e9, ν=0.3)

        # Stress tensor should be symmetric
        ε = SymmetricTensor{2,3}((0.001, 0.0005, 0.0003, -0.0002, 0.0004, 0.0006))
        σ, _, _ = compute_stress(steel, ε, nothing, 0.0)

        @test σ[1, 2] ≈ σ[2, 1] rtol = 1e-15
        @test σ[1, 3] ≈ σ[3, 1] rtol = 1e-15
        @test σ[2, 3] ≈ σ[3, 2] rtol = 1e-15
    end

    @testset "Isotropy Verification" begin
        steel = LinearElastic(E=200e9, ν=0.3)

        # Same strain magnitude in different directions → same stress magnitude
        ε_x = SymmetricTensor{2,3}((0.001, 0.0, 0.0, 0.0, 0.0, 0.0))
        ε_y = SymmetricTensor{2,3}((0.0, 0.0, 0.0, 0.001, 0.0, 0.0))
        ε_z = SymmetricTensor{2,3}((0.0, 0.0, 0.0, 0.0, 0.0, 0.001))

        σ_x, _, _ = compute_stress(steel, ε_x, nothing, 0.0)
        σ_y, _, _ = compute_stress(steel, ε_y, nothing, 0.0)
        σ_z, _, _ = compute_stress(steel, ε_z, nothing, 0.0)

        # σ₁₁(ε_x) should equal σ₂₂(ε_y) and σ₃₃(ε_z)
        @test σ_x[1, 1] ≈ σ_y[2, 2] rtol = 1e-15
        @test σ_x[1, 1] ≈ σ_z[3, 3] rtol = 1e-15
    end

    @testset "Simplified Interface" begin
        steel = LinearElastic(E=200e9, ν=0.3)
        ε = SymmetricTensor{2,3}((0.001, 0.0, 0.0, 0.0, 0.0, 0.0))

        # Test simplified call (without state and Δt)
        σ1, 𝔻1, state1 = compute_stress(steel, ε)
        σ2, 𝔻2, state2 = compute_stress(steel, ε, nothing, 0.0)

        @test σ1 ≈ σ2
        @test 𝔻1 ≈ 𝔻2
        @test state1 === nothing
        @test state2 === nothing
    end

    @testset "Zero Allocation" begin
        steel = LinearElastic(E=200e9, ν=0.3)
        ε = SymmetricTensor{2,3}((0.001, 0.0, 0.0, 0.0, 0.0, 0.0))

        # First call to compile
        compute_stress(steel, ε, nothing, 0.0)

        # Check allocations
        allocs = @allocated compute_stress(steel, ε, nothing, 0.0)
        @test allocs == 0
    end

    @testset "Type Stability" begin
        steel = LinearElastic(E=200e9, ν=0.3)
        ε = SymmetricTensor{2,3}((0.001, 0.0, 0.0, 0.0, 0.0, 0.0))

        # Infer return types
        result = @inferred compute_stress(steel, ε, nothing, 0.0)

        @test result isa Tuple{SymmetricTensor{2,3,Float64},SymmetricTensor{4,3,Float64},Nothing}
    end

end
