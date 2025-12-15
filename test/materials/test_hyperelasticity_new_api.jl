"""
# Hyperelasticity - NEW API (Test-Driven Development)

**What:** Shows how hyperelastic materials SHOULD work with the NEW API

**Why:**
- **Finite strain** - Large deformations (rubber, soft tissue, biomechanics)
- **Energy-based** - Strain energy function Ψ(F)
- **Frame-invariant** - Material objectivity (rotation independence)
- **Multiple models** - Neo-Hookean, Mooney-Rivlin, Ogden, etc.
- **Incompressibility** - Nearly incompressible (ν ≈ 0.5)

**NEW API Concepts:**
1. **Hyperelastic material types** - Neo-Hookean, Mooney-Rivlin, Ogden
2. **Strain energy function** - Ψ(F) and derivatives
3. **Push-forward stress** - σ = (1/J) P F^T (Cauchy from 1st Piola-Kirchhoff)
4. **Tangent moduli** - C_ijkl = ∂²Ψ/∂F_ij∂F_kl
5. **Incompressibility constraint** - det(F) ≈ 1

**Test Problems:**

## Test 1: Neo-Hookean (Simplest Hyperelastic)
- Ψ = μ/2 (I₁ - 3) - μ ln(J) + λ/2 (ln J)²
- Validates stress computation from energy
- Tests incompressibility limit

## Test 2: Mooney-Rivlin (Two-Parameter)
- Ψ = C₁(I₁ - 3) + C₂(I₂ - 3) + κ/2 (J - 1)²
- Better for rubber than Neo-Hookean
- Validates second invariant I₂

## Test 3: Ogden Model (Multi-Term)
- Ψ = Σᵢ μᵢ/αᵢ (λ₁^αᵢ + λ₂^αᵢ + λ₃^αᵢ - 3)
- Uses principal stretches λᵢ
- Most accurate for rubber

## Test 4: Uniaxial Tension Test
- Compare to experimental data
- Validates material parameters
- Tests large strain (λ > 2)

**Expected Behavior (when implemented):**
✅ Stress computed from ∂Ψ/∂F correctly
✅ Tangent moduli symmetric and positive-definite
✅ Incompressibility enforced (J ≈ 1)
✅ Frame-invariant (rotations don't change Ψ)
✅ Matches experimental stress-strain curves
✅ Works with Newton-Krylov solver

**Status:** 🚧 VISIONARY TEST - Implementation in progress
"""

using Test
using JuliaFEM
using Tensors
using LinearAlgebra
using Statistics

@testset "Hyperelasticity - NEW API (TDD)" begin

    # =============================================================================
    # NEO-HOOKEAN MODEL (SIMPLEST)
    # =============================================================================

    @testset "Neo-Hookean Material (Visionary)" begin
        @test_skip begin  # Skip until implemented

            # Material parameters
            μ = 1000.0    # Shear modulus
            λ = 2000.0    # Lame parameter (nearly incompressible)

            # NEW: Hyperelastic material type
            material = NeoHookean(
                μ=μ,
                λ=λ,
                formulation=:compressible  # or :incompressible
            )

            # Test deformation gradient
            F = Tensor{2,3}((
                1.2, 0.1, 0.0,
                0.0, 0.9, 0.0,
                0.0, 0.0, 1.0
            ))

            # Compute strain energy
            Ψ = strain_energy(material, F)

            # Neo-Hookean energy:
            # Ψ = μ/2 (I₁ - 3) - μ ln(J) + λ/2 (ln J)²
            C = tdot(F)  # Right Cauchy-Green: C = F^T F
            I₁ = tr(C)    # First invariant
            J = det(F)    # Volume ratio

            Ψ_analytical = μ / 2 * (I₁ - 3) - μ * log(J) + λ / 2 * log(J)^2

            @test isapprox(Ψ, Ψ_analytical, rtol=1e-10)

            # First Piola-Kirchhoff stress: P = ∂Ψ/∂F
            P = first_piola_kirchhoff_stress(material, F)

            # Analytical P for Neo-Hookean
            F_inv = inv(F)
            P_analytical = μ * (F - tdot(F_inv)) + λ * log(J) * tdot(F_inv)

            @test P ≈ P_analytical rtol = 1e-10

            # Cauchy stress: σ = (1/J) P F^T
            σ = cauchy_stress(material, F)
            σ_from_P = (1 / J) * P ⊡ transpose(F)

            @test σ ≈ σ_from_P rtol = 1e-10

        end
    end

    # =============================================================================
    # INCOMPRESSIBILITY CONSTRAINT
    # =============================================================================

    @testset "Incompressibility (Nearly) (Visionary)" begin
        @test_skip begin

            # Nearly incompressible (ν → 0.5)
            E = 1000.0
            ν = 0.499  # Nearly incompressible

            μ = E / (2 * (1 + ν))
            λ = E * ν / ((1 + ν) * (1 - 2ν))  # Very large!

            material = NeoHookean(
                μ=μ,
                λ=λ,
                formulation=:compressible
            )

            # Deformation (should preserve volume)
            F = Tensor{2,3}((
                1.5, 0.0, 0.0,
                0.0, 1 / sqrt(1.5), 0.0,
                0.0, 0.0, 1 / sqrt(1.5)
            ))

            J = det(F)

            # For incompressible: J = 1 (volume preserving)
            @test isapprox(J, 1.0, atol=1e-3)

            # Hydrostatic pressure enforces incompressibility
            σ = cauchy_stress(material, F)
            p = -tr(σ) / 3  # Hydrostatic pressure

            # For nearly incompressible, pressure should be large
            @test abs(p) > 1000.0  # Significant pressure

        end
    end

    # =============================================================================
    # MOONEY-RIVLIN MODEL
    # =============================================================================

    @testset "Mooney-Rivlin Material (Visionary)" begin
        @test_skip begin

            # Material parameters (typical for rubber)
            C₁ = 0.5  # MPa
            C₂ = 0.1  # MPa
            κ = 100.0  # Bulk modulus (incompressibility)

            # NEW: Mooney-Rivlin material
            material = MooneyRivlin(
                C1=C₁,
                C2=C₂,
                bulk_modulus=κ
            )

            # Test deformation
            F = Tensor{2,3}((
                1.5, 0.2, 0.0,
                0.1, 0.8, 0.0,
                0.0, 0.0, 1.0
            ))

            # Strain energy: Ψ = C₁(I₁ - 3) + C₂(I₂ - 3) + κ/2 (J - 1)²
            C = tdot(F)
            I₁ = tr(C)
            I₂ = 0.5 * (tr(C)^2 - tr(C ⊡ C))  # Second invariant
            J = det(F)

            Ψ = strain_energy(material, F)
            Ψ_analytical = C₁ * (I₁ - 3) + C₂ * (I₂ - 3) + κ / 2 * (J - 1)^2

            @test isapprox(Ψ, Ψ_analytical, rtol=1e-10)

            # Stress computation
            P = first_piola_kirchhoff_stress(material, F)
            σ = cauchy_stress(material, F)

            # Validate symmetry of Cauchy stress
            @test isapprox(σ, transpose(σ), atol=1e-10)

        end
    end

    # =============================================================================
    # OGDEN MODEL (PRINCIPAL STRETCHES)
    # =============================================================================

    @testset "Ogden Material (Visionary)" begin
        @test_skip begin

            # Ogden parameters (multi-term)
            μ_terms = [1.0, 0.5, 0.2]    # Shear moduli
            α_terms = [2.0, 3.0, -2.0]   # Exponents
            κ = 100.0

            # NEW: Ogden material
            material = Ogden(
                mu=μ_terms,
                alpha=α_terms,
                bulk_modulus=κ
            )

            # Deformation
            F = Tensor{2,3}((
                2.0, 0.0, 0.0,
                0.0, 0.6, 0.0,
                0.0, 0.0, 0.8
            ))

            # Compute principal stretches
            C = tdot(F)
            eigenvalues_C = eigvals(C)
            λ = sqrt.(eigenvalues_C)  # Principal stretches

            # Strain energy: Ψ = Σᵢ μᵢ/αᵢ (λ₁^αᵢ + λ₂^αᵢ + λ₃^αᵢ - 3)
            Ψ = strain_energy(material, F)

            J = det(F)
            Ψ_analytical = sum(
                μ_terms[i] / α_terms[i] * (sum(λ .^ α_terms[i]) - 3)
                for i in 1:3
            ) + κ / 2 * (J - 1)^2

            @test isapprox(Ψ, Ψ_analytical, rtol=1e-10)

        end
    end

    # =============================================================================
    # UNIAXIAL TENSION TEST
    # =============================================================================

    @testset "Uniaxial Tension (Large Strain) (Visionary)" begin
        @test_skip begin

            # Geometry: Unit cube under tension
            mesh = generate_mesh(
                geometry=UnitCube(),
                element_type=Hex8,
                n_elements=(4, 4, 4)
            )

            # Material (Neo-Hookean rubber)
            material = NeoHookean(
                μ=1.0,    # MPa
                λ=10.0,   # Nearly incompressible
                density=1000.0
            )

            # Physics
            elastic_physics = ContinuumPhysics{Displacement}(
                material=material,
                formulation=FullThreeD(),
                finite_strain=true  # CRITICAL!
            )

            domain = Domain(
                name="RUBBER",
                elements=mesh.elements,
                physics=elastic_physics
            )

            # Boundary conditions
            bc_fixed = DirichletBC(
                nodes=mesh.node_sets["LEFT"],
                dof=:displacement,
                values=[0.0, 0.0, 0.0]
            )

            # Applied stretch (λ = 2.0, 100% strain!)
            u_applied = 1.0  # Stretch from 1.0 to 2.0

            bc_stretch = DirichletBC(
                nodes=mesh.node_sets["RIGHT"],
                dof=:displacement,
                component=:x,
                value=u_applied
            )

            # Nonlinear problem (finite strain)
            problem = NonlinearProblem(
                domains=[domain],
                boundary_conditions=[bc_fixed, bc_stretch]
            )

            # Newton-Krylov solver
            solver = NewtonKrylovSolver(
                max_iterations=20,
                convergence_tol=1e-6,
                krylov_solver=GMRES(restart=30),
                line_search=BacktrackingLineSearch()
            )

            solution = solve!(problem, solver)

            # Extract stress-stretch curve
            λ = 1.0 + u_applied  # Stretch ratio

            # Compute engineering stress
            F_avg = compute_average_deformation_gradient(solution)
            P_avg = first_piola_kirchhoff_stress(material, F_avg)

            # Engineering stress: σ_eng = P_11 (1st Piola-Kirchhoff in x)
            σ_eng = P_avg[1, 1]

            # For Neo-Hookean uniaxial:
            # σ_eng = μ(λ - 1/λ²)
            σ_analytical = material.μ * (λ - 1 / λ^2)

            @test isapprox(σ_eng, σ_analytical, rtol=0.05)  # 5% tolerance

        end
    end

    # =============================================================================
    # FRAME INVARIANCE (OBJECTIVITY)
    # =============================================================================

    @testset "Frame Invariance (Visionary)" begin
        @test_skip begin

            material = NeoHookean(μ=1000.0, λ=2000.0)

            # Deformation gradient
            F = Tensor{2,3}((
                1.2, 0.1, 0.0,
                0.0, 0.9, 0.0,
                0.0, 0.0, 1.0
            ))

            # Rotation tensor (90° about z-axis)
            θ = π / 2
            Q = Tensor{2,3}((
                cos(θ), -sin(θ), 0.0,
                sin(θ), cos(θ), 0.0,
                0.0, 0.0, 1.0
            ))

            # Rotated deformation: F' = Q F
            F_rotated = Q ⊡ F

            # Strain energy should be INVARIANT
            Ψ = strain_energy(material, F)
            Ψ_rotated = strain_energy(material, F_rotated)

            @test isapprox(Ψ, Ψ_rotated, rtol=1e-10)

            # Cauchy stress should transform: σ' = Q σ Q^T
            σ = cauchy_stress(material, F)
            σ_rotated = cauchy_stress(material, F_rotated)

            σ_transformed = Q ⊡ σ ⊡ transpose(Q)

            @test σ_rotated ≈ σ_transformed rtol = 1e-10

        end
    end

    # =============================================================================
    # TANGENT MODULI (FOR NEWTON)
    # =============================================================================

    @testset "Tangent Moduli (Visionary)" begin
        @test_skip begin

            material = NeoHookean(μ=1000.0, λ=2000.0)

            F = Tensor{2,3}((
                1.2, 0.1, 0.0,
                0.0, 0.9, 0.0,
                0.0, 0.0, 1.0
            ))

            # Material tangent: C_ijkl = ∂²Ψ/∂F_ij∂F_kl
            C = material_tangent(material, F)

            # Validate major symmetry: C_ijkl = C_klij
            # (Minor symmetries don't hold for finite strain)
            for i in 1:3, j in 1:3, k in 1:3, l in 1:3
                @test isapprox(C[i, j, k, l], C[k, l, i, j], atol=1e-10)
            end

            # Validate positive-definiteness
            # For small perturbation δF, δ²Ψ = C_ijkl δF_ij δF_kl > 0
            δF = 0.01 * rand(Tensor{2,3})
            δ²Ψ = dcontract(dcontract(C, δF), δF)

            @test δ²Ψ > 0  # Positive-definite

        end
    end

    # =============================================================================
    # PSEUDO-CODE: HYPERELASTIC ASSEMBLY
    # =============================================================================

    @testset "Hyperelastic Assembly Pattern (Visionary)" begin
        # Pseudo-code showing finite strain assembly

        println("\n" * "="^70)
        println("HYPERELASTIC ASSEMBLY (FINITE STRAIN)")
        println("="^70)

        assembly_pseudo = """
        # For hyperelastic materials, assembly uses current configuration

        function compute_residual_hyperelastic(u, material, elements)
            r = zeros(length(u))
            
            for elem in elements
                # Current deformation gradient: F = I + ∇u
                X = reference_coordinates(elem)  # Undeformed
                x = X + u[elem.nodes]            # Deformed
                
                for ip in integration_points(elem)
                    # Jacobian in reference config
                    J₀, dN_dX = jacobian(elem, ip, X)
                    
                    # Deformation gradient: F = ∂x/∂X
                    F = compute_deformation_gradient(x, dN_dX)
                    
                    # 1st Piola-Kirchhoff stress: P = ∂Ψ/∂F
                    P = first_piola_kirchhoff_stress(material, F)
                    
                    # Residual: r = ∫_Ω₀ P : ∇_X(δu) dΩ₀
                    # = ∫_Ω₀ P_iJ (dN_I/dX_J) dΩ₀
                    for I in 1:n_nodes
                        for i in 1:3
                            for J in 1:3
                                r[3*(I-1)+i] += P[i,J] * dN_dX[I,J] * J₀ * ip.weight
                            end
                        end
                    end
                end
            end
            
            return r
        end

        # Tangent stiffness (for Newton):
        # K_t = ∫_Ω₀ (dN_I/dX_K) C_iJkL (dN_J/dX_L) dΩ₀
        # where C_iJkL = ∂²Ψ/∂F_iJ∂F_kL
        """

        println(assembly_pseudo)
        println("="^70)
        println("✓ Uses reference configuration Ω₀ (not current!)")
        println("✓ Deformation gradient F = ∂x/∂X")
        println("✓ 1st Piola-Kirchhoff stress P = ∂Ψ/∂F")
        println("✓ Tangent moduli C = ∂²Ψ/∂F²")
        println("✓ Works with nodal assembly (same pattern!)")
        println("="^70)
    end

    # =============================================================================
    # KEY ARCHITECTURAL INSIGHTS
    # =============================================================================

    println("\n" * "="^70)
    println("HYPERELASTICITY ARCHITECTURE INSIGHTS (NEW API)")
    println("="^70)
    println("✓ Hyperelastic materials: NeoHookean, MooneyRivlin, Ogden")
    println("✓ Strain energy function Ψ(F) is fundamental")
    println("✓ Stress from energy: P = ∂Ψ/∂F, σ = (1/J) P F^T")
    println("✓ Tangent from energy: C = ∂²Ψ/∂F²")
    println("✓ Frame-invariant: Rotations don't change Ψ")
    println("✓ Incompressibility: det(F) ≈ 1 for rubber")
    println("✓ Works with Newton-Krylov (unsymmetric OK)")
    println("✓ Assembly in reference config (not current!)")
    println("✓ Large strains: λ > 2 (100%+ strain)")
    println("="^70)

end

"""
# IMPLEMENTATION NOTES

## Hyperelastic Material Models

### Neo-Hookean (Simplest)

**Strain energy:**
Ψ = μ/2 (I₁ - 3) - μ ln(J) + λ/2 (ln J)²

where:
- I₁ = tr(C) = tr(F^T F) (first invariant)
- J = det(F) (volume ratio)
- μ, λ = Lame parameters

**1st Piola-Kirchhoff stress:**
P = ∂Ψ/∂F = μ(F - F^{-T}) + λ ln(J) F^{-T}

where F^{-T} = (F^{-1})^T.

**Cauchy stress:**
σ = (1/J) P F^T = μ/J (B - I) + λ/J ln(J) I

where B = F F^T (left Cauchy-Green).

**Use case:** Rubber at moderate strains (< 100%).

### Mooney-Rivlin (Two-Parameter)

**Strain energy:**
Ψ = C₁(I₁ - 3) + C₂(I₂ - 3) + κ/2 (J - 1)²

where:
- I₁ = tr(C)
- I₂ = 1/2 [(tr C)² - tr(C²)]
- J = det(F)
- C₁, C₂ = material parameters
- κ = bulk modulus

**Better fit for rubber** than Neo-Hookean.

**Relation to Neo-Hookean:** C₂ = 0 → Neo-Hookean.

### Ogden Model (Multi-Term)

**Strain energy:**
Ψ = Σᵢ μᵢ/αᵢ (λ₁^{αᵢ} + λ₂^{αᵢ} + λ₃^{αᵢ} - 3) + κ/2 (J - 1)²

where:
- λ₁, λ₂, λ₃ = principal stretches (eigenvalues of F)
- μᵢ, αᵢ = material parameters (typically 3-6 terms)

**Most accurate for rubber** (fits experimental data well).

**Implementation:** Requires spectral decomposition of C.

## Stress Measures

### First Piola-Kirchhoff (P)

**Definition:** P = ∂Ψ/∂F

**Properties:**
- Non-symmetric
- Force per reference area
- Work-conjugate to F

**Use:** Weak form in reference config.

### Cauchy Stress (σ)

**Definition:** σ = (1/J) P F^T

**Properties:**
- Symmetric
- True stress (force per current area)
- What we measure

**Use:** Post-processing, failure criteria.

### Second Piola-Kirchhoff (S)

**Definition:** S = F^{-1} P = J F^{-1} σ F^{-T}

**Properties:**
- Symmetric
- Work-conjugate to E (Green-Lagrange strain)
- Energy-conjugate

**Use:** Theoretical derivations.

## Incompressibility

**Constraint:** det(F) = J = 1 (volume preserving)

**Nearly incompressible:** ν → 0.5, λ → ∞

**Enforcement:**

1. **Penalty:** Add κ/2 (J - 1)² to Ψ (large κ)
2. **Lagrange multiplier:** Introduce pressure p
3. **Mixed formulation:** (u, p) unknowns

**JuliaFEM approach:** Penalty for compressible materials, mixed for truly incompressible.

## Frame Invariance (Objectivity)

**Definition:** Strain energy invariant under rigid rotations.

**Mathematical:** Ψ(Q F) = Ψ(F) for all rotations Q.

**Why:** Material doesn't "know" about global rotations.

**Implementation:** Use invariants (I₁, I₂, I₃) or principal stretches (λᵢ).

**Validation:**
```julia
Q = rotation_matrix(θ)
F_rotated = Q * F
@test strain_energy(F_rotated) ≈ strain_energy(F)
```

## Tangent Moduli

**Material tangent:**
C_{iJkL} = ∂²Ψ/∂F_{iJ}∂F_{kL}

**Spatial tangent:**
c_{ijkl} = (1/J) F_{iI} F_{jJ} F_{kK} F_{lL} C_{IJKL}

**Symmetries:**
- Major: C_{iJkL} = C_{kLiJ} (from Ψ)
- Minor: Generally NOT symmetric in hyperelasticity

**Use in Newton:**
δP = C : δF

## Assembly (Finite Strain)

**Residual (weak form):**
r = ∫_{Ω₀} P : ∇_X(δu) dΩ₀ - f_ext

**In components:**
r_I^i = ∫_{Ω₀} P_{iJ} ∂N_I/∂X_J dΩ₀ - f_I^i

**Tangent stiffness:**
K_{IJ}^{ik} = ∫_{Ω₀} ∂N_I/∂X_K C_{iJkL} ∂N_J/∂X_L dΩ₀

**Key differences from small strain:**
1. Integrate over Ω₀ (reference), not Ω (current)
2. Use ∂N/∂X (reference gradients), not ∂N/∂x
3. P (1st Piola-Kirchhoff), not σ (Cauchy)

## Nodal Assembly (Hyperelastic)

```julia
function tangent_matvec_hyperelastic!(w, v, u_current, material, elements, node_to_elements)
    Threads.@threads for node_i in 1:n_nodes
        w_local = zero(Vec{3})
        
        for elem in node_to_elements[node_i]
            # Current deformation
            F = deformation_gradient(elem, u_current)
            
            # Material tangent
            C = material_tangent(material, F)
            
            for node_j in elem.nodes
                # Tangent block: K_t_ij
                K_t_ij = compute_hyperelastic_tangent_block(elem, node_i, node_j, C, F)
                
                v_j = Vec{3}(v[3*(node_j-1)+1:3*node_j])
                w_local += K_t_ij ⊡ v_j
            end
        end
        
        w[3*(node_i-1)+1:3*node_i] = w_local
    end
end
```

**Same pattern as linear elasticity!** Only the tangent computation changes.

## Next Steps

1. Implement `NeoHookean` material type
2. Implement `strain_energy` function
3. Implement `first_piola_kirchhoff_stress`
4. Implement `cauchy_stress` (push-forward)
5. Implement `material_tangent` (C_ijkl)
6. Implement `MooneyRivlin` material
7. Implement `Ogden` material (spectral decomposition)
8. Validate against analytical solutions
9. Validate against experimental data
10. Performance benchmarks

"""
