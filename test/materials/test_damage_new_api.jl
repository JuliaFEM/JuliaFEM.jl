"""
# Damage Mechanics - NEW API (Test-Driven Development)

**What:** Shows how damage models SHOULD work with the NEW API

**Why:**
- **Stiffness degradation** - Material weakens (cracks, voids)
- **Irreversible** - Damage cannot heal (unlike plasticity unloading)
- **Mesh independence** - Regularization required (crack band, nonlocal)
- **Failure prediction** - Crack initiation, propagation

**NEW API Concepts:**
1. **Damage variable** - d ∈ [0,1] where 0=intact, 1=failed
2. **Effective stress** - σ̄ = σ/(1-d) (undamaged configuration)
3. **Damage evolution** - ḋ = f(ε, ε_max, damage parameters)
4. **Regularization** - Length scale to avoid mesh sensitivity
5. **Coupled damage-plasticity** - Combined degradation + permanent deformation

**Test Problems:**

## Test 1: Isotropic Damage
- Stiffness reduction: E_eff = (1-d) E
- Damage driven by strain energy
- Validates crack initiation

## Test 2: Ductile Damage (Lemaitre)
- Coupled damage-plasticity
- Damage from plastic dissipation
- Validates void growth

## Test 3: Crack Band Regularization
- Mesh-independent energy dissipation
- Length scale h = element size
- Validates objectivity

## Test 4: Damage Unloading
- Permanent stiffness loss
- No damage healing
- Validates irreversibility

**Expected Behavior (when implemented):**
✅ Damage variable d ∈ [0,1]
✅ Stiffness degrades smoothly
✅ Mesh-independent fracture energy
✅ No healing on unload
✅ Crack localization captured
✅ Failure criterion satisfied

**Status:** 🚧 VISIONARY TEST - Implementation in progress
"""

using Test
using JuliaFEM
using Tensors
using LinearAlgebra
using Statistics

@testset "Damage Mechanics - NEW API (TDD)" begin

    # =============================================================================
    # ISOTROPIC DAMAGE
    # =============================================================================

    @testset "Isotropic Damage - Strain-Based (Visionary)" begin
        @test_skip begin  # Skip until implemented

            # Material parameters
            E = 200e3      # Young's modulus (MPa)
            ν = 0.3        # Poisson's ratio
            ε_d0 = 0.001   # Damage threshold strain
            ε_f = 0.01     # Failure strain

            # NEW: Isotropic damage material
            material = IsotropicDamage(
                E=E,
                ν=ν,
                damage_threshold=ε_d0,
                failure_strain=ε_f,
                evolution_law=:exponential  # or :linear, :power
            )

            # Strain history (uniaxial tension)
            ε_max = 0.015  # Beyond failure
            n_steps = 200
            ε_history = range(0, ε_max, length=n_steps)

            σ_history = []
            d_history = []  # Damage variable

            # Internal state
            state = DamageState(
                d=0.0,           # Damage variable (0=intact, 1=failed)
                ε_eq_max=0.0     # Maximum equivalent strain (history)
            )

            for ε in ε_history
                # Strain tensor (uniaxial tension)
                ε_tensor = SymmetricTensor{2,3}((
                    ε, 0.0, 0.0,
                    0.0, -ν * ε, 0.0,
                    0.0, 0.0, -ν * ε
                ))

                # Compute stress (with damage)
                σ, state_new = compute_stress_damage(material, ε_tensor, state)

                push!(σ_history, σ[1, 1])
                push!(d_history, state_new.d)

                state = state_new
            end

            # Validate elastic region (no damage)
            elastic_indices = findall(ε_history .<= ε_d0)
            for i in elastic_indices
                @test isapprox(σ_history[i], E * ε_history[i], rtol=0.01)
                @test d_history[i] == 0.0
            end

            # Validate damage growth
            damage_indices = findall((ε_history .> ε_d0) .& (ε_history .< ε_f))
            for i in damage_indices
                @test 0.0 < d_history[i] < 1.0

                # Effective stiffness reduces
                E_eff = E * (1 - d_history[i])
                @test E_eff < E
            end

            # Validate failure
            failure_indices = findall(ε_history .>= ε_f)
            for i in failure_indices
                @test d_history[i] >= 0.99  # Nearly complete damage
                @test σ_history[i] < 0.1 * maximum(σ_history)  # Stress vanishes
            end

        end
    end

    # =============================================================================
    # DAMAGE EVOLUTION LAWS
    # =============================================================================

    @testset "Damage Evolution Laws (Visionary)" begin
        @test_skip begin

            E = 200e3
            ν = 0.3
            ε_d0 = 0.001
            ε_f = 0.01

            # Test different evolution laws
            laws = [:linear, :exponential, :power]

            for law in laws
                material = IsotropicDamage(
                    E=E,
                    ν=ν,
                    damage_threshold=ε_d0,
                    failure_strain=ε_f,
                    evolution_law=law
                )

                state = DamageState(d=0.0, ε_eq_max=0.0)

                # Strain at 50% between threshold and failure
                ε_mid = (ε_d0 + ε_f) / 2
                ε_tensor = SymmetricTensor{2,3}((ε_mid, 0.0, 0.0, 0.0, 0.0, 0.0))

                σ, state_new = compute_stress_damage(material, ε_tensor, state)

                # Damage should be growing
                @test 0.0 < state_new.d < 1.0

                # Different laws give different d values
                println("Law: $law, d = $(state_new.d)")
            end

        end
    end

    # =============================================================================
    # DUCTILE DAMAGE (LEMAITRE MODEL)
    # =============================================================================

    @testset "Ductile Damage - Coupled Plasticity (Visionary)" begin
        @test_skip begin

            # Coupled damage-plasticity
            material = DuctileDamage(
                # Elastic properties
                E=200e3,
                ν=0.3,

                # Plasticity
                yield_stress=250.0,
                hardening=IsotropicHardening(H=2000.0),

                # Damage (Lemaitre)
                damage_threshold=0.001,  # Plastic strain threshold
                S_crit=1.0,              # Critical damage value
                s_damage=1.5,            # Triaxiality sensitivity
                damage_exponent=2.0
            )

            # Strain history (tension to failure)
            ε_max = 0.05
            n_steps = 200
            ε_history = range(0, ε_max, length=n_steps)

            σ_history = []
            d_history = []
            ε_p_history = []

            state = DuctileDamageState(
                ε_p=zero(SymmetricTensor{2,3}),
                ε_p_eq=0.0,
                d=0.0,
                α=zero(SymmetricTensor{2,3})
            )

            for ε in ε_history
                ε_tensor = SymmetricTensor{2,3}((ε, 0.0, 0.0, 0.0, 0.0, 0.0))
                σ, state = compute_stress_ductile_damage(material, ε_tensor, state)

                push!(σ_history, σ[1, 1])
                push!(d_history, state.d)
                push!(ε_p_history, state.ε_p_eq)
            end

            # Validate coupling: Damage grows with plastic strain
            @test all(diff(d_history[ε_p_history.>material.damage_threshold]) .>= 0)

            # Validate softening: Peak stress followed by descent
            σ_max_idx = argmax(σ_history)
            @test σ_max_idx < length(σ_history)  # Not at end

            # After peak, stress decreases (softening)
            @test σ_history[end] < σ_history[σ_max_idx]

            # Damage increases monotonically
            @test all(diff(d_history) .>= 0)

        end
    end

    # =============================================================================
    # CRACK BAND REGULARIZATION
    # =============================================================================

    @testset "Crack Band Regularization (Visionary)" begin
        @test_skip begin

            # Fracture energy per unit area (N/mm)
            G_f = 0.1  # Fracture energy

            # Two different mesh sizes
            h_coarse = 10.0  # mm
            h_fine = 2.0     # mm

            # Crack band materials (adjust ε_f based on mesh)
            # Energy = G_f = ∫ σ dε * h
            # For linear softening: G_f ≈ (1/2) σ_max ε_f * h

            σ_max = 10.0  # Tensile strength (MPa)

            # Coarse mesh: larger ε_f
            ε_f_coarse = 2 * G_f / (σ_max * h_coarse)
            material_coarse = IsotropicDamage(
                E=200e3,
                ν=0.3,
                damage_threshold=σ_max / 200e3,
                failure_strain=ε_f_coarse,
                evolution_law=:linear,
                crack_band_width=h_coarse
            )

            # Fine mesh: smaller ε_f
            ε_f_fine = 2 * G_f / (σ_max * h_fine)
            material_fine = IsotropicDamage(
                E=200e3,
                ν=0.3,
                damage_threshold=σ_max / 200e3,
                failure_strain=ε_f_fine,
                evolution_law=:linear,
                crack_band_width=h_fine
            )

            # Compute energy dissipation for both
            function compute_dissipation(material, ε_max, n_steps)
                ε_history = range(0, ε_max, length=n_steps)
                σ_history = []

                state = DamageState(d=0.0, ε_eq_max=0.0)

                for ε in ε_history
                    ε_tensor = SymmetricTensor{2,3}((ε, 0.0, 0.0, 0.0, 0.0, 0.0))
                    σ, state = compute_stress_damage(material, ε_tensor, state)
                    push!(σ_history, σ[1, 1])
                end

                # Integrate σ dε (trapezoid rule)
                W = sum((σ_history[i] + σ_history[i+1]) / 2 * (ε_history[i+1] - ε_history[i])
                        for i in 1:length(ε_history)-1)

                return W
            end

            W_coarse = compute_dissipation(material_coarse, 1.2 * ε_f_coarse, 500)
            W_fine = compute_dissipation(material_fine, 1.2 * ε_f_fine, 500)

            # Fracture energy per volume
            G_v_coarse = W_coarse * h_coarse
            G_v_fine = W_fine * h_fine

            # Should be mesh-independent!
            @test isapprox(G_v_coarse, G_v_fine, rtol=0.1)
            @test isapprox(G_v_coarse, G_f, rtol=0.1)

        end
    end

    # =============================================================================
    # DAMAGE UNLOADING (IRREVERSIBILITY)
    # =============================================================================

    @testset "Damage Unloading - Irreversible (Visionary)" begin
        @test_skip begin

            material = IsotropicDamage(
                E=200e3,
                ν=0.3,
                damage_threshold=0.001,
                failure_strain=0.01,
                evolution_law=:linear
            )

            # Load-unload cycle
            ε_max = 0.005  # Partial damage

            # Loading
            ε_loading = range(0, ε_max, length=100)
            σ_loading = []
            d_loading = []

            state = DamageState(d=0.0, ε_eq_max=0.0)

            for ε in ε_loading
                ε_tensor = SymmetricTensor{2,3}((ε, 0.0, 0.0, 0.0, 0.0, 0.0))
                σ, state = compute_stress_damage(material, ε_tensor, state)

                push!(σ_loading, σ[1, 1])
                push!(d_loading, state.d)
            end

            d_max = state.d  # Damage at peak load

            # Unloading
            ε_unloading = range(ε_max, 0, length=100)
            σ_unloading = []
            d_unloading = []

            for ε in ε_unloading
                ε_tensor = SymmetricTensor{2,3}((ε, 0.0, 0.0, 0.0, 0.0, 0.0))
                σ, state = compute_stress_damage(material, ε_tensor, state)

                push!(σ_unloading, σ[1, 1])
                push!(d_unloading, state.d)
            end

            # Validate irreversibility
            @test all(d_unloading .≈ d_max)  # Damage does not heal!

            # Validate reduced stiffness
            E_damaged = (1 - d_max) * material.E

            # Unloading slope should match damaged stiffness
            # (linear regression on unloading curve)
            ε_unload_vals = collect(ε_unloading)
            slope = (σ_unloading[1] - σ_unloading[end]) / (ε_unload_vals[1] - ε_unload_vals[end])

            @test isapprox(slope, E_damaged, rtol=0.1)

        end
    end

    # =============================================================================
    # CONSISTENT TANGENT (DAMAGE)
    # =============================================================================

    @testset "Consistent Tangent - Damage (Visionary)" begin
        @test_skip begin

            material = IsotropicDamage(
                E=200e3,
                ν=0.3,
                damage_threshold=0.001,
                failure_strain=0.01,
                evolution_law=:exponential
            )

            # Strain state (damaged)
            ε = SymmetricTensor{2,3}((0.003, 0.001, 0.0, 0.001, 0.002, 0.0))
            state = DamageState(d=0.3, ε_eq_max=0.003)

            # Compute stress and tangent
            σ, state_new, C_damage = compute_stress_tangent_damage(material, ε, state)

            # Validate tangent via finite difference
            δε = 1e-8

            for i in 1:6  # Voigt notation
                ε_pert = ε + δε * basis_symmetric_tensor(i)
                σ_pert, _ = compute_stress_damage(material, ε_pert, state)

                dσ_numerical = (σ_pert - σ) / δε
                dσ_tangent = C_damage ⊡ basis_symmetric_tensor(i)

                @test isapprox(dσ_numerical, dσ_tangent, rtol=0.01)
            end

            # Validate symmetry
            for i in 1:6, j in 1:6
                @test isapprox(C_damage[i, j], C_damage[j, i], atol=1e-10)
            end

            # Validate degradation
            C_elastic = compute_elastic_stiffness(material)

            # Damaged stiffness should be less
            @test norm(C_damage) < norm(C_elastic)

        end
    end

    # =============================================================================
    # PSEUDO-CODE: DAMAGE INTEGRATION
    # =============================================================================

    @testset "Damage Integration Pattern (Visionary)" begin
        # Pseudo-code showing damage evolution

        println("\n" * "="^70)
        println("DAMAGE INTEGRATION")
        println("="^70)

        integration_pseudo = """
        # Damage evolution (strain-based isotropic)

        function compute_stress_damage(material, ε, state_old)
            # 1. Compute equivalent strain
            ε_eq = compute_equivalent_strain(ε)  # e.g., sqrt(ε:ε)
            
            # 2. Update history (loading surface)
            ε_eq_max = max(state_old.ε_eq_max, ε_eq)
            
            # 3. Check damage threshold
            if ε_eq_max <= material.ε_d0
                # No damage
                d = 0.0
            else
                # Damage evolution
                d = compute_damage(material, ε_eq_max)
            end
            
            # 4. Effective stress
            # Strain energy equivalence: W = W̄
            # σ : ε = σ̄ : ε in undamaged configuration
            
            # Elastic stress (undamaged)
            C_elastic = compute_elastic_stiffness(material)
            σ_undamaged = C_elastic ⊡ ε
            
            # Apply damage
            σ = (1 - d) * σ_undamaged
            
            # 5. Update state
            state_new = DamageState(d, ε_eq_max)
            
            return σ, state_new
        end

        # Damage evolution laws
        function compute_damage(material, ε_eq_max)
            ε_d0 = material.damage_threshold
            ε_f = material.failure_strain
            
            if material.evolution_law == :linear
                # Linear: d = (ε - ε_d0) / (ε_f - ε_d0)
                d = (ε_eq_max - ε_d0) / (ε_f - ε_d0)
                
            elseif material.evolution_law == :exponential
                # Exponential: d = 1 - exp(-α(ε - ε_d0))
                α = -log(0.01) / (ε_f - ε_d0)  # d(ε_f) ≈ 0.99
                d = 1 - exp(-α * (ε_eq_max - ε_d0))
                
            elseif material.evolution_law == :power
                # Power law: d = ((ε - ε_d0)/(ε_f - ε_d0))^n
                n = 2.0
                d = ((ε_eq_max - ε_d0) / (ε_f - ε_d0))^n
            end
            
            return clamp(d, 0.0, 0.99)  # Numerical: never fully failed
        end
        """

        println(integration_pseudo)
        println("="^70)
        println("✓ Equivalent strain: History variable")
        println("✓ Loading surface: ε_eq_max = max(ε_eq_max_old, ε_eq)")
        println("✓ Damage evolution: d = f(ε_eq_max)")
        println("✓ Effective stress: σ = (1-d) σ_undamaged")
        println("✓ Irreversible: d never decreases")
        println("="^70)
    end

    # =============================================================================
    # KEY ARCHITECTURAL INSIGHTS
    # =============================================================================

    println("\n" * "="^70)
    println("DAMAGE MECHANICS ARCHITECTURE INSIGHTS (NEW API)")
    println("="^70)
    println("✓ Isotropic damage: Stiffness degradation (1-d)E")
    println("✓ Damage variable: d ∈ [0,1] (0=intact, 1=failed)")
    println("✓ Irreversible: d never decreases (no healing)")
    println("✓ History: ε_eq_max (maximum strain ever reached)")
    println("✓ Evolution laws: Linear, exponential, power")
    println("✓ Crack band: Mesh-independent G_f via length scale h")
    println("✓ Ductile damage: Coupled with plasticity (Lemaitre)")
    println("✓ Consistent tangent: C_damage = ∂σ/∂ε (with damage)")
    println("✓ Regularization: REQUIRED for mesh objectivity")
    println("✓ Works with Newton-Krylov (tangent from damage law)")
    println("="^70)

end

"""
# IMPLEMENTATION NOTES

## Isotropic Damage

### Damage Variable

**Definition:** d ∈ [0,1]
- d = 0: Intact material
- d = 1: Completely damaged (failed)

**Effective stress concept:**
σ̄ = σ / (1 - d)

where σ̄ = stress in undamaged (effective) configuration.

**Strain energy equivalence:**
W(σ, ε) = W̄(σ̄, ε)

Implies:
σ = (1 - d) σ̄

where σ̄ = C_elastic : ε.

### Equivalent Strain

**For isotropic damage:** Need scalar measure of strain state.

**Tension-driven:**
ε_eq = √(<ε_1>² + <ε_2>² + <ε_3>²)

where <·> = positive part, ε_i = principal strains.

**Reason:** Damage in tension (cracks open), not compression.

**Alternative (modified von Mises):**
ε_eq = κ I_1 / (1-2ν) + √(3J_2) / (1+ν)

where κ weighs volumetric vs deviatoric.

### Damage Evolution Laws

**Linear:**
d = (ε_eq - ε_d0) / (ε_f - ε_d0)  for ε_eq ∈ [ε_d0, ε_f]

**Exponential (smoother):**
d = 1 - exp(-α(ε_eq - ε_d0))

where α chosen such that d(ε_f) ≈ 0.99.

**Power law:**
d = ((ε_eq - ε_d0) / (ε_f - ε_d0))^n

where n controls softening rate.

### History Variable

**Loading surface:** ε_eq_max = max(ε_eq_history)

**Damage depends on history:**
d = f(ε_eq_max)  NOT f(ε_eq)

**Irreversibility:** ε_eq_max only increases.

**Update:**
```julia
ε_eq_max_new = max(ε_eq_max_old, ε_eq_current)
```

## Crack Band Regularization

### Mesh Sensitivity Problem

**Without regularization:** Fracture energy depends on mesh size!

G_num = ∫ σ dε * h

where h = element size.

**Finer mesh → less energy dissipation → spurious brittleness.**

### Crack Band Model

**Idea:** Fracture happens over a band of width h.

**Energy balance:**
G_f = ∫_0^{ε_f} σ dε * h

where G_f = fracture energy per unit area (material property).

**Adjust failure strain:**
ε_f = G_f / (∫_0^{ε_f} σ dε * h)

**For linear softening:**
ε_f = 2 G_f / (σ_max h)

where σ_max = tensile strength.

**Result:** Mesh-independent fracture energy!

### Implementation

```julia
struct IsotropicDamage
    E::Float64
    ν::Float64
    damage_threshold::Float64
    failure_strain::Float64  # Computed from G_f and h!
    crack_band_width::Float64  # h (element size)
end

function IsotropicDamage(; E, ν, G_f, σ_max, h)
    ε_d0 = σ_max / E
    ε_f = ε_d0 + 2*G_f / (σ_max * h)  # Linear softening
    
    return IsotropicDamage(E, ν, ε_d0, ε_f, h)
end
```

## Ductile Damage (Lemaitre Model)

### Coupling: Damage + Plasticity

**Damage drives plasticity:**
σ_y_eff = σ_y / (1 - d)

**Plasticity drives damage:**
ḋ = f(plastic dissipation)

### Lemaitre Damage Evolution

**Damage rate:**
ḋ = (Y / S)^s ε̇_p_eq

where:
- Y = damage energy release rate = (σ_eq²) / (2E(1-d)²)
- S = material damage strength
- s = damage exponent
- ε̇_p_eq = equivalent plastic strain rate

**Damage threshold:**
d = 0 until ε_p_eq > ε_p_threshold

**Triaxiality influence:**
Y = Y(σ_eq, σ_m / σ_eq)

where σ_m = mean stress (pressure).

**High triaxiality → void growth → more damage.**

### Integration

```julia
function compute_ductile_damage(material, σ, ε_p_eq, state)
    if ε_p_eq < material.ε_p_threshold
        return 0.0
    end
    
    # Damage energy release rate
    σ_eq = von_mises_stress(σ)
    Y = σ_eq^2 / (2 * material.E * (1 - state.d)^2)
    
    # Triaxiality (optional)
    σ_m = trace(σ) / 3
    η = σ_m / σ_eq
    
    # Damage increment
    Δε_p = ε_p_eq - state.ε_p_eq_old
    Δd = (Y / material.S)^material.s * Δε_p
    
    d_new = state.d + Δd
    
    return clamp(d_new, 0.0, 0.99)
end
```

## Consistent Tangent (Damage)

**For Newton:** Need C_damage = dσ/dε.

**Elastic damage:**
σ = (1 - d) C_elastic : ε

**Tangent:**
C_damage = (1 - d) C_elastic + ∂d/∂ε ⊗ σ_elastic

where ⊗ = outer product.

**Derivative of damage:**
∂d/∂ε = (∂d/∂ε_eq) (∂ε_eq/∂ε)

**Chain rule through damage evolution law.**

**For exponential:**
∂d/∂ε_eq = α exp(-α(ε_eq - ε_d0))

**For linear:**
∂d/∂ε_eq = 1 / (ε_f - ε_d0)

### Symmetry

**Major symmetry:** C_damage may NOT be symmetric if ∂d/∂ε ⊗ σ not symmetric.

**Options:**
1. Symmetrize: C_sym = (C + C^T) / 2
2. Use unsymmetric solver (GMRES handles it!)

## Internal State Storage

**Per integration point:**
```julia
struct DamageState
    d::Float64              # Damage variable
    ε_eq_max::Float64       # Maximum equivalent strain (history)
end

# Coupled damage-plasticity
struct DuctileDamageState{dim}
    ε_p::SymmetricTensor{2,dim}
    ε_p_eq::Float64
    d::Float64
    α::SymmetricTensor{2,dim}
end
```

**Element-level:**
```julia
struct DamageElement
    topology::AbstractTopology
    basis::AbstractBasis
    nodes::NTuple{N,Int}
    state::Vector{DamageState}  # Per integration point
end
```

## Nodal Assembly (Damage)

```julia
function tangent_matvec_damage!(w, v, u_current, material, elements, states)
    Threads.@threads for node_i in 1:n_nodes
        w_local = zero(Vec{3})
        
        for elem in node_to_elements[node_i]
            for (ip_idx, ip) in enumerate(integration_points(elem))
                # Current state
                state = states[elem][ip_idx]
                
                # Strain
                ε = compute_strain(elem, ip, u_current)
                
                # Consistent tangent (with damage)
                σ, state_new, C_damage = compute_stress_tangent_damage(material, ε, state)
                
                for node_j in elem.nodes
                    # Tangent block (damaged stiffness)
                    K_t_ij = compute_damage_tangent_block(elem, node_i, node_j, C_damage, ip)
                    
                    v_j = Vec{3}(v[3*(node_j-1)+1:3*node_j])
                    w_local += K_t_ij ⊡ v_j
                end
                
                # Update state
                states[elem][ip_idx] = state_new
            end
        end
        
        w[3*(node_i-1)+1:3*node_i] = w_local
    end
end
```

**Key:** Damage state updated each iteration!

## Regularization Techniques

### 1. Crack Band (Local)

**Pros:** Simple, fast
**Cons:** Still some mesh sensitivity

### 2. Nonlocal Damage

**Averaged equivalent strain:**
ε̄_eq(x) = (1/V_R) ∫_{B_R(x)} α(||y-x||) ε_eq(y) dy

where:
- B_R(x) = ball of radius R around x
- α = weight function (Gaussian)

**Damage driven by ε̄_eq instead of ε_eq.**

**Pros:** Mesh-independent
**Cons:** Expensive (nonlocal averaging)

### 3. Gradient Damage

**Higher-order PDE:**
ε̄_eq - c ∇²ε̄_eq = ε_eq

where c = internal length scale.

**Requires additional DOF or coupled system.**

**Pros:** Mesh-independent, smooth localization
**Cons:** Complex implementation

### 4. Phase Field (Future)

**Crack as diffuse interface:**
φ(x) ∈ [0,1] where φ=1 is crack.

**Coupled:**
- Elasticity with φ-dependent stiffness
- Allen-Cahn or Ginzburg-Landau equation for φ

**Pros:** Arbitrary crack topology, no remeshing
**Cons:** Very expensive

## Next Steps

1. Implement `IsotropicDamage` material type
2. Implement `DamageState` struct
3. Implement damage evolution laws
4. Implement crack band regularization
5. Implement `DuctileDamage` (coupled)
6. Implement consistent tangent
7. Validate mesh independence
8. Validate against experiments
9. Performance benchmarks

"""
