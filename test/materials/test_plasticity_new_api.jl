"""
# Plasticity - NEW API (Test-Driven Development)

**What:** Shows how plasticity models SHOULD work with the NEW API

**Why:**
- **Permanent deformation** - Irreversible (metals, soils)
- **Yield criterion** - von Mises, Tresca, Drucker-Prager
- **Hardening** - Isotropic, kinematic, mixed
- **Rate-independence** - Path-independent (classical plasticity)
- **History-dependent** - Internal state variables

**NEW API Concepts:**
1. **Plastic material types** - J2Plasticity, DruckerPrager
2. **Yield function** - f(σ, α) ≤ 0 (elastic domain)
3. **Flow rule** - Plastic strain rate direction
4. **Hardening laws** - Isotropic (expanding yield surface), kinematic (translation)
5. **Return mapping** - Radial return, closest point projection

**Test Problems:**

## Test 1: J2 Plasticity (von Mises)
- Yield: f = √(3J₂) - σ_y(ε_p)
- Isotropic hardening
- Validates return mapping algorithm

## Test 2: Kinematic Hardening
- Backstress α (yield surface translates)
- Armstrong-Frederick model
- Validates ratcheting behavior

## Test 3: Perfect Plasticity
- No hardening: σ_y = constant
- Validates elastic-perfectly plastic
- Tests limit load

## Test 4: Cyclic Loading (Bauschinger Effect)
- Load → Unload → Reverse load
- Validates kinematic hardening
- Tests hysteresis loop

**Expected Behavior (when implemented):**
✅ Yield criterion correctly evaluated
✅ Elastic-plastic split accurate
✅ Return mapping converges
✅ Hardening modulus computed correctly
✅ Consistent tangent for Newton
✅ Path-independence validated

**Status:** 🚧 VISIONARY TEST - Implementation in progress
"""

using Test
using JuliaFEM
using Tensors
using LinearAlgebra
using Statistics

@testset "Plasticity - NEW API (TDD)" begin

    # =============================================================================
    # J2 PLASTICITY (VON MISES)
    # =============================================================================

    @testset "J2 Plasticity - Isotropic Hardening (Visionary)" begin
        @test_skip begin  # Skip until implemented

            # Material parameters
            E = 200e3     # Young's modulus (MPa)
            ν = 0.3       # Poisson's ratio
            σ_y0 = 250.0  # Initial yield stress (MPa)
            H = 2000.0    # Hardening modulus (MPa)

            # NEW: J2 plasticity material
            material = J2Plasticity(
                E=E,
                ν=ν,
                yield_stress=σ_y0,
                hardening=IsotropicHardening(H=H),
                hardening_law=:linear  # or :exponential, :voce
            )

            # Strain history (uniaxial tension)
            ε_max = 0.005  # 0.5% total strain
            n_steps = 100
            ε_history = range(0, ε_max, length=n_steps)

            # Strain tensor (uniaxial)
            σ_history = []
            ε_p_history = []

            # Internal state
            state = PlasticState(
                ε_p=zero(SymmetricTensor{2,3}),  # Plastic strain
                ε_p_eq=0.0,                      # Equivalent plastic strain
                α=zero(SymmetricTensor{2,3})     # Backstress (if kinematic)
            )

            for ε in ε_history
                # Strain tensor (uniaxial tension in x)
                ε_total = SymmetricTensor{2,3}((
                    ε, 0.0, 0.0,
                    0.0, -ν * ε, 0.0,
                    0.0, 0.0, -ν * ε
                ))

                # Compute stress (with return mapping)
                σ, state_new = compute_stress(material, ε_total, state)

                push!(σ_history, σ[1, 1])  # Axial stress
                push!(ε_p_history, state_new.ε_p_eq)

                state = state_new
            end

            # Validate elastic region
            ε_elastic = σ_y0 / E
            elastic_indices = findall(ε_history .<= ε_elastic)

            for i in elastic_indices
                # Elastic: σ = E ε
                @test isapprox(σ_history[i], E * ε_history[i], rtol=0.01)
                @test ε_p_history[i] == 0.0
            end

            # Validate plastic region
            plastic_indices = findall(ε_history .> ε_elastic)

            for i in plastic_indices
                # Plastic: σ_y(ε_p) = σ_y0 + H ε_p
                ε_p = ε_p_history[i]
                σ_y_current = σ_y0 + H * ε_p

                # Stress should be at yield
                @test isapprox(σ_history[i], σ_y_current, rtol=0.01)
            end

        end
    end

    # =============================================================================
    # RETURN MAPPING ALGORITHM
    # =============================================================================

    @testset "Radial Return Mapping (Visionary)" begin
        @test_skip begin

            material = J2Plasticity(
                E=200e3,
                ν=0.3,
                yield_stress=250.0,
                hardening=IsotropicHardening(H=2000.0)
            )

            # Trial elastic step (exceed yield)
            ε_trial = SymmetricTensor{2,3}((
                0.003, 0.001, 0.0,
                0.001, 0.002, 0.0,
                0.0, 0.0, 0.0
            ))

            state = PlasticState(
                ε_p=zero(SymmetricTensor{2,3}),
                ε_p_eq=0.0,
                α=zero(SymmetricTensor{2,3})
            )

            # Elastic predictor
            σ_trial = elastic_stress(material, ε_trial - state.ε_p)

            # Yield function
            s_trial = dev(σ_trial)  # Deviatoric stress
            q_trial = sqrt(1.5 * dcontract(s_trial, s_trial))  # von Mises stress

            f_trial = q_trial - material.yield_stress

            if f_trial > 0
                # Plastic: Return mapping required
                σ, state_new = return_mapping(material, σ_trial, state)

                # Validate yield criterion satisfied
                s = dev(σ)
                q = sqrt(1.5 * dcontract(s, s))
                σ_y_current = material.yield_stress + material.H * state_new.ε_p_eq

                @test isapprox(q, σ_y_current, atol=1e-6)

                # Validate plastic strain increased
                @test state_new.ε_p_eq > state.ε_p_eq

            else
                # Elastic: No return mapping
                @test f_trial <= 0
            end

        end
    end

    # =============================================================================
    # KINEMATIC HARDENING (ARMSTRONG-FREDERICK)
    # =============================================================================

    @testset "Kinematic Hardening (Visionary)" begin
        @test_skip begin

            # Material with kinematic hardening
            material = J2Plasticity(
                E=200e3,
                ν=0.3,
                yield_stress=250.0,
                hardening=KinematicHardening(
                    C=5000.0,  # Kinematic hardening modulus
                    γ=50.0     # Armstrong-Frederick parameter
                ),
                mixed_hardening=false
            )

            # Cyclic loading: tension → compression
            ε_max = 0.005
            n_cycles = 3

            ε_history = []
            σ_history = []

            state = PlasticState(
                ε_p=zero(SymmetricTensor{2,3}),
                ε_p_eq=0.0,
                α=zero(SymmetricTensor{2,3})  # Backstress
            )

            for cycle in 1:n_cycles
                # Tension
                for ε in range(0, ε_max, length=50)
                    ε_tensor = SymmetricTensor{2,3}((ε, 0.0, 0.0, 0.0, 0.0, 0.0))
                    σ, state = compute_stress(material, ε_tensor, state)

                    push!(ε_history, ε)
                    push!(σ_history, σ[1, 1])
                end

                # Compression
                for ε in range(ε_max, -ε_max, length=100)
                    ε_tensor = SymmetricTensor{2,3}((ε, 0.0, 0.0, 0.0, 0.0, 0.0))
                    σ, state = compute_stress(material, ε_tensor, state)

                    push!(ε_history, ε)
                    push!(σ_history, σ[1, 1])
                end

                # Back to tension
                for ε in range(-ε_max, 0, length=50)
                    ε_tensor = SymmetricTensor{2,3}((ε, 0.0, 0.0, 0.0, 0.0, 0.0))
                    σ, state = compute_stress(material, ε_tensor, state)

                    push!(ε_history, ε)
                    push!(σ_history, σ[1, 1])
                end
            end

            # Validate Bauschinger effect
            # Yield stress in compression < initial yield
            σ_y_compression = minimum(σ_history[ε_history.<0])
            @test abs(σ_y_compression) < material.yield_stress

            # Validate hysteresis loop closes
            # (For stabilized cycle)
            @test length(ε_history) > 0

        end
    end

    # =============================================================================
    # PERFECT PLASTICITY (NO HARDENING)
    # =============================================================================

    @testset "Perfect Plasticity (Visionary)" begin
        @test_skip begin

            # No hardening: H = 0
            material = J2Plasticity(
                E=200e3,
                ν=0.3,
                yield_stress=250.0,
                hardening=NoHardening()  # H = 0
            )

            # Large strain (well into plastic)
            ε_max = 0.01  # 1% strain
            ε_history = range(0, ε_max, length=100)

            σ_history = []
            state = PlasticState(
                ε_p=zero(SymmetricTensor{2,3}),
                ε_p_eq=0.0,
                α=zero(SymmetricTensor{2,3})
            )

            for ε in ε_history
                ε_tensor = SymmetricTensor{2,3}((ε, 0.0, 0.0, 0.0, 0.0, 0.0))
                σ, state = compute_stress(material, ε_tensor, state)
                push!(σ_history, σ[1, 1])
            end

            # After yield, stress should be constant
            ε_yield = material.yield_stress / material.E
            plastic_indices = findall(ε_history .> ε_yield)

            σ_plastic = σ_history[plastic_indices]

            # All plastic stresses ≈ σ_y (no hardening!)
            @test all(isapprox.(σ_plastic, material.yield_stress, rtol=0.01))

        end
    end

    # =============================================================================
    # CONSISTENT TANGENT (FOR NEWTON)
    # =============================================================================

    @testset "Consistent Tangent (Visionary)" begin
        @test_skip begin

            material = J2Plasticity(
                E=200e3,
                ν=0.3,
                yield_stress=250.0,
                hardening=IsotropicHardening(H=2000.0)
            )

            # Strain state (plastic)
            ε = SymmetricTensor{2,3}((0.003, 0.001, 0.0, 0.001, 0.002, 0.0))
            state = PlasticState(
                ε_p=SymmetricTensor{2,3}((0.001, 0.0, 0.0, 0.0, 0.0, 0.0)),
                ε_p_eq=0.001,
                α=zero(SymmetricTensor{2,3})
            )

            # Compute stress and tangent
            σ, state_new, C_ep = compute_stress_tangent(material, ε, state)

            # Validate tangent via finite difference
            δε = 1e-8

            for i in 1:6  # Voigt notation
                ε_pert = ε + δε * basis_symmetric_tensor(i)
                σ_pert, _ = compute_stress(material, ε_pert, state)

                dσ_numerical = (σ_pert - σ) / δε
                dσ_tangent = C_ep ⊡ basis_symmetric_tensor(i)

                @test isapprox(dσ_numerical, dσ_tangent, rtol=0.01)
            end

            # Validate symmetry (major)
            for i in 1:6, j in 1:6
                @test isapprox(C_ep[i, j], C_ep[j, i], atol=1e-10)
            end

        end
    end

    # =============================================================================
    # MULTI-AXIAL LOADING
    # =============================================================================

    @testset "Multi-Axial Loading (Visionary)" begin
        @test_skip begin

            material = J2Plasticity(
                E=200e3,
                ν=0.3,
                yield_stress=250.0,
                hardening=IsotropicHardening(H=2000.0)
            )

            # Combined tension + shear
            ε_axial_max = 0.003
            ε_shear_max = 0.002

            n_steps = 100
            σ_history = []

            state = PlasticState(
                ε_p=zero(SymmetricTensor{2,3}),
                ε_p_eq=0.0,
                α=zero(SymmetricTensor{2,3})
            )

            for i in 1:n_steps
                # Proportional loading
                ε_axial = ε_axial_max * i / n_steps
                ε_shear = ε_shear_max * i / n_steps

                ε = SymmetricTensor{2,3}((
                    ε_axial, ε_shear, 0.0,
                    ε_shear, 0.0, 0.0,
                    0.0, 0.0, 0.0
                ))

                σ, state = compute_stress(material, ε, state)
                push!(σ_history, σ)
            end

            # Validate von Mises yield criterion
            for σ in σ_history
                s = dev(σ)
                q = sqrt(1.5 * dcontract(s, s))
                σ_y_current = material.yield_stress + material.H * state.ε_p_eq

                # Should be at or below yield
                @test q <= σ_y_current + 1e-6
            end

        end
    end

    # =============================================================================
    # PSEUDO-CODE: PLASTICITY INTEGRATION
    # =============================================================================

    @testset "Plasticity Integration Pattern (Visionary)" begin
        # Pseudo-code showing return mapping

        println("\n" * "="^70)
        println("PLASTICITY INTEGRATION (RETURN MAPPING)")
        println("="^70)

        integration_pseudo = """
        # Return mapping algorithm (radial return for J2)

        function compute_stress_plastic(material, ε_total, state_old)
            # 1. Elastic predictor
            ε_elastic_trial = ε_total - state_old.ε_p
            σ_trial = C_elastic ⊡ ε_elastic_trial
            
            # 2. Check yield
            s_trial = dev(σ_trial)  # Deviatoric
            q_trial = sqrt(1.5 * s_trial : s_trial)  # von Mises
            
            σ_y = material.σ_y0 + H * state_old.ε_p_eq
            f_trial = q_trial - σ_y
            
            if f_trial <= 0
                # Elastic: Accept trial state
                return σ_trial, state_old
            end
            
            # 3. Plastic corrector (return mapping)
            # Solve for Δλ (plastic multiplier)
            # f = q - σ_y(ε_p + Δλ) = 0
            
            # Newton iteration
            Δλ = 0.0
            for iter in 1:max_iter
                σ_y_current = material.σ_y0 + H * (state_old.ε_p_eq + Δλ)
                q_current = q_trial - 3*G*Δλ  # G = shear modulus
                
                f = q_current - σ_y_current
                
                if abs(f) < tol
                    break
                end
                
                # Derivative: df/dΔλ
                df_dΔλ = -3*G - H
                
                # Update
                Δλ -= f / df_dΔλ
            end
            
            # 4. Update stress and state
            n = s_trial / norm(s_trial)  # Flow direction
            
            σ = σ_trial - 2*G*Δλ * n
            ε_p_new = state_old.ε_p + Δλ * n
            ε_p_eq_new = state_old.ε_p_eq + Δλ
            
            state_new = PlasticState(ε_p_new, ε_p_eq_new, state_old.α)
            
            return σ, state_new
        end
        """

        println(integration_pseudo)
        println("="^70)
        println("✓ Elastic predictor: Assume elastic step")
        println("✓ Check yield: f(σ_trial) ≤ 0?")
        println("✓ Return mapping: Project back to yield surface")
        println("✓ Newton iteration: Solve for plastic multiplier Δλ")
        println("✓ Update state: ε_p, ε_p_eq, α")
        println("="^70)
    end

    # =============================================================================
    # KEY ARCHITECTURAL INSIGHTS
    # =============================================================================

    println("\n" * "="^70)
    println("PLASTICITY ARCHITECTURE INSIGHTS (NEW API)")
    println("="^70)
    println("✓ J2 plasticity: von Mises yield, isotropic/kinematic hardening")
    println("✓ Yield function: f(σ, α) = √(3J₂) - σ_y(ε_p)")
    println("✓ Return mapping: Radial return, closest point projection")
    println("✓ Consistent tangent: C_ep for Newton quadratic convergence")
    println("✓ Internal state: ε_p, ε_p_eq, α (per integration point!)")
    println("✓ Isotropic hardening: Yield surface expands")
    println("✓ Kinematic hardening: Yield surface translates (Bauschinger)")
    println("✓ Path-independent: Same final state for same strain path")
    println("✓ Works with Newton-Krylov (tangent from return mapping)")
    println("="^70)

end

"""
# IMPLEMENTATION NOTES

## J2 Plasticity (von Mises)

### Yield Function

**Definition:**
f(σ, ε_p) = √(3J₂) - σ_y(ε_p)

where:
- J₂ = (1/2) s:s (second deviatoric invariant)
- s = σ - (1/3)tr(σ)I (deviatoric stress)
- σ_y(ε_p) = yield stress (function of plastic strain)

**Equivalent form:**
f = q - σ_y

where q = √(3J₂) = von Mises stress.

**Elastic domain:** f ≤ 0

**Yield surface:** f = 0

### Flow Rule

**Associative plasticity:** Plastic strain rate direction = yield gradient

ε̇_p = λ̇ ∂f/∂σ = λ̇ (3/2) s/q = λ̇ n

where:
- λ̇ = plastic multiplier (rate)
- n = (3/2) s/q = flow direction (unit deviatoric)

**Properties:**
- Incompressible: tr(ε̇_p) = 0 (volume preserving)
- Radial: ε̇_p ∝ s (proportional to deviatoric stress)

### Hardening Laws

**Isotropic (linear):**
σ_y(ε_p) = σ_y0 + H ε_p_eq

where:
- σ_y0 = initial yield stress
- H = hardening modulus
- ε_p_eq = ∫ √(2/3 ε̇_p:ε̇_p) dt = equivalent plastic strain

**Isotropic (exponential/Voce):**
σ_y(ε_p) = σ_∞ - (σ_∞ - σ_y0) exp(-b ε_p_eq)

Saturates to σ_∞.

**Kinematic (Armstrong-Frederick):**
α̇ = C ε̇_p - γ α λ̇

where:
- α = backstress (2nd order tensor)
- C = kinematic hardening modulus
- γ = recall parameter

**Modified yield:**
f = √(3/2 (s-α):(s-α)) - σ_y

### Return Mapping Algorithm

**Problem:** Given ε_{n+1}, find σ_{n+1} and state_{n+1}.

**Elastic predictor:**
```
ε_e_trial = ε_{n+1} - ε_p_n
σ_trial = C_elastic : ε_e_trial
```

**Check yield:**
```
f_trial = q_trial - σ_y(ε_p_eq_n)
```

**If f_trial ≤ 0:** Elastic, return (σ_trial, state_n)

**If f_trial > 0:** Plastic, solve for Δλ:

**Consistency condition:**
f(σ_{n+1}, ε_p_eq_{n+1}) = 0

**Discretized flow rule:**
ε_p_{n+1} = ε_p_n + Δλ n

**Stress update:**
σ_{n+1} = σ_trial - 2G Δλ n

where G = shear modulus.

**Yield condition:**
q_{n+1} = q_trial - 3G Δλ = σ_y(ε_p_eq_n + Δλ)

**Solve for Δλ (Newton):**
```julia
function return_mapping(σ_trial, state, material)
    s_trial = dev(σ_trial)
    q_trial = sqrt(1.5 * dcontract(s_trial, s_trial))
    n = s_trial / norm(s_trial)
    
    # Initial guess
    Δλ = 0.0
    ε_p_eq_old = state.ε_p_eq
    G = material.E / (2*(1 + material.ν))
    H = material.H
    
    for iter in 1:max_iter
        # Current yield stress
        σ_y = material.σ_y0 + H * (ε_p_eq_old + Δλ)
        
        # Residual
        f = q_trial - 3*G*Δλ - σ_y
        
        if abs(f) < tol
            break
        end
        
        # Derivative
        df_dΔλ = -3*G - H
        
        # Newton update
        Δλ -= f / df_dΔλ
    end
    
    # Update stress
    σ = σ_trial - 2*G*Δλ * n
    
    # Update state
    ε_p_new = state.ε_p + Δλ * n
    ε_p_eq_new = ε_p_eq_old + Δλ
    
    return σ, PlasticState(ε_p_new, ε_p_eq_new, state.α)
end
```

### Consistent Tangent

**For Newton convergence:** Need C_ep = dσ/dε (algorithmic tangent).

**Elastic:**
C_ep = C_elastic

**Plastic:** More complex!

C_ep = C_elastic - (2G)² / (3G + H) * (n ⊗ n)

where ⊗ = outer product.

**Derivation:** Chain rule through return mapping.

**Properties:**
- Symmetric (major symmetry)
- Positive-definite (for H > 0)
- Converges to C_elastic as Δλ → 0

## Internal State Storage

**Per integration point:**
```julia
struct PlasticState{dim}
    ε_p::SymmetricTensor{2,dim}      # Plastic strain
    ε_p_eq::Float64                  # Equivalent plastic strain
    α::SymmetricTensor{2,dim}        # Backstress (kinematic)
end
```

**Element-level:**
```julia
struct PlasticElement
    topology::AbstractTopology
    basis::AbstractBasis
    nodes::NTuple{N,Int}
    state::Vector{PlasticState}  # One per integration point!
end
```

**Key:** State is HISTORY-DEPENDENT, must be stored!

## Nodal Assembly (Plasticity)

```julia
function tangent_matvec_plastic!(w, v, u_current, material, elements, states)
    Threads.@threads for node_i in 1:n_nodes
        w_local = zero(Vec{3})
        
        for elem in node_to_elements[node_i]
            for (ip_idx, ip) in enumerate(integration_points(elem))
                # Current state at this integration point
                state = states[elem][ip_idx]
                
                # Strain
                ε = compute_strain(elem, ip, u_current)
                
                # Consistent tangent (elastic or plastic)
                σ, state_new, C_ep = compute_stress_tangent(material, ε, state)
                
                for node_j in elem.nodes
                    # Tangent block
                    K_t_ij = compute_plastic_tangent_block(elem, node_i, node_j, C_ep, ip)
                    
                    v_j = Vec{3}(v[3*(node_j-1)+1:3*node_j])
                    w_local += K_t_ij ⊡ v_j
                end
                
                # Update state (for next iteration)
                states[elem][ip_idx] = state_new
            end
        end
        
        w[3*(node_i-1)+1:3*node_i] = w_local
    end
end
```

**Key:** State updated during tangent computation!

## Next Steps

1. Implement `J2Plasticity` material type
2. Implement `PlasticState` struct
3. Implement `return_mapping` algorithm
4. Implement `compute_stress_plastic`
5. Implement `consistent_tangent_plastic`
6. Implement hardening laws (isotropic, kinematic)
7. Implement state storage (per integration point)
8. Validate against analytical solutions
9. Validate against experimental data
10. Performance benchmarks

"""
