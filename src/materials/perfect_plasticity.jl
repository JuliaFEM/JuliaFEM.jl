"""
Perfect Plasticity Material (J2 Plasticity with Kinematic Hardening)

Classical von Mises plasticity with radial return mapping algorithm.
"""

using Tensors
using LinearAlgebra

# PlasticityState struct removed - now using compositional NamedTuple
# State is represented as: (ε_p=SymmetricTensor{2,3}, α=SymmetricTensor{2,3}, κ=Float64)
# This compositional design is inferred automatically from material traits

"""
    PerfectPlasticity <: AbstractPlasticMaterial

J2 (von Mises) plasticity with kinematic hardening.

# Fields
- `E::Float64` - Young's modulus [Pa]
- `ν::Float64` - Poisson's ratio [-]
- `σ_y::Float64` - Yield stress [Pa]
- `H::Float64` - Hardening modulus [Pa]
"""
struct PerfectPlasticity <: AbstractPlasticMaterial
    E::Float64   # Young's modulus [Pa]
    ν::Float64   # Poisson's ratio [-]
    σ_y::Float64 # Yield stress [Pa]
    H::Float64   # Hardening modulus [Pa]

    # Derived properties (for performance)
    μ::Float64   # Shear modulus
    λ::Float64   # Lamé parameter

    function PerfectPlasticity(E::Float64, ν::Float64, σ_y::Float64, H::Float64)
        # Validate inputs
        E > 0.0 || throw(ArgumentError("Young's modulus must be positive, got E = $E"))
        -1.0 < ν < 0.5 || throw(ArgumentError("Poisson's ratio must satisfy -1 < ν < 0.5, got ν = $ν"))
        σ_y > 0.0 || throw(ArgumentError("Yield stress must be positive, got σ_y = $σ_y"))
        H ≥ 0.0 || throw(ArgumentError("Hardening modulus must be non-negative, got H = $H"))

        # Compute Lamé parameters
        μ = E / (2(1 + ν))
        λ = E * ν / ((1 + ν) * (1 - 2ν))

        new(E, ν, σ_y, H, μ, λ)
    end
end

"""
    PerfectPlasticity(; E, ν, σ_y, H)

Keyword constructor for perfect plasticity material.
"""
PerfectPlasticity(; E::Real, ν::Real, σ_y::Real, H::Real) =
    PerfectPlasticity(Float64(E), Float64(ν), Float64(σ_y), Float64(H))

# Trait declaration: PerfectPlasticity has strain-dependent tangent and state
material_behavior(::PerfectPlasticity) = StatefulStrainDependent()

# New trait system: Physics and state variable requirements
# PerfectPlasticity supports 3D elasticity with J2 plasticity
supported_physics(::PerfectPlasticity) = (Elasticity{3}(),)

# PerfectPlasticity requires three state variables (compositional design)
required_state_variables(::PerfectPlasticity) = (PlasticStrain, Backstress, EquivalentPlasticStrain)

"""
    compute_stress(material::PerfectPlasticity, ε, state_old, Δt) -> (σ, 𝔻, state_new)

Compute stress and consistent tangent using radial return mapping.
"""
function compute_stress(material::PerfectPlasticity,
    ε::SymmetricTensor{2,3},
    state_old::NamedTuple=NamedTuple(),
    Δt::Float64=0.0)
    # Extract material parameters
    μ = material.μ
    λ = material.λ
    σ_y = material.σ_y
    H = material.H

    # Extract old state (with defaults for initial/empty state)
    ε_p_old = get(state_old, :ε_p, zero(SymmetricTensor{2,3}))
    α_old = get(state_old, :α, zero(SymmetricTensor{2,3}))
    κ_old = get(state_old, :κ, 0.0)

    # Elastic strain
    ε_e = ε - ε_p_old

    # STEP 1: Elastic Predictor
    # σ_trial = λ·tr(ε_e)·I + 2μ·ε_e
    I = one(ε)
    σ_trial = λ * tr(ε_e) * I + 2μ * ε_e

    # STEP 2: Check Yield Criterion
    # Deviatoric part of relative stress
    s_trial = dev(σ_trial - α_old)

    # Von Mises equivalent stress
    s_trial_norm = √(3 / 2) * √(s_trial ⊡ s_trial)  # ||s||

    # Yield function
    f_trial = s_trial_norm - σ_y

    # STEP 3: Plastic Corrector or Return
    if f_trial ≤ 0.0
        # ==================== ELASTIC ====================
        σ = σ_trial
        state_new = (ε_p=ε_p_old, α=α_old, κ=κ_old)  # No state change

        # Elastic tangent
        𝔻 = λ * I ⊗ I + 2μ * symmetric_identity_tensor()

    else
        # ==================== PLASTIC ====================
        # Flow direction (unit deviatoric tensor)
        n = s_trial / s_trial_norm

        # Plastic multiplier (closed-form solution for J2 plasticity with kinematic hardening)
        # Derivation: After return mapping:
        #   dev(σ - α_new) = dev(σ_trial - 2μΔλn - α_old - (2/3)HΔλn)
        #                  = s_trial - (2μ + 2H/3)Δλn  (since dev(n) = n)
        # Yield criterion: √(3/2)||dev(σ - α_new)|| = σ_y
        # Since n is parallel to s_trial:
        #   √(3/2)(||s_trial|| - (2μ + 2H/3)Δλ) = σ_y
        #   √(3/2)||s_trial|| - σ_y = √(3/2)(2μ + 2H/3)Δλ
        #   f_trial = √(3/2)(2μ + 2H/3)Δλ
        #   Δλ = f_trial / (√(3/2)(2μ + 2H/3))
        #   Δλ = f_trial / (√(3/2) * 2(3μ + H)/3)
        #   Δλ = 3f_trial / (2√(3/2)(3μ + H))
        #   Δλ = 3f_trial / (2(3μ + H)/√(3/2))
        #   Δλ = 3f_trial * √(3/2) / (2(3μ + H))
        # Simplifying: √(3/2) * 3/2 = √(27/8) = 3√3/(2√8) = 3√3/(4√2) = 3/(2√(2/3))
        # But cleaner: Δλ = f_trial / ((2μ + 2H/3))
        Δλ = f_trial / (2μ + (2.0 / 3.0) * H)

        # Update stress (radial return) - before backstress!
        σ = σ_trial - 2μ * Δλ * n

        # Update backstress (kinematic hardening) - must use same n
        α_new = α_old + (2.0 / 3.0) * H * Δλ * n

        # Update plastic strain
        ε_p_new = ε_p_old + Δλ * n

        # Update equivalent plastic strain
        κ_new = κ_old + Δλ

        # New state (compositional NamedTuple)
        state_new = (ε_p=ε_p_new, α=α_new, κ=κ_new)

        # Consistent tangent (elastoplastic)
        # For kinematic hardening: 𝔻^ep = 𝔻^e - (4μ²/(2μ + 2H/3)) · (n ⊗ n)
        𝔻_e = λ * I ⊗ I + 2μ * symmetric_identity_tensor()

        # Algorithmic tangent (consistent with return mapping)
        𝔻 = 𝔻_e - (4μ^2 / (2μ + (2.0 / 3.0) * H)) * (n ⊗ n)
    end

    return σ, 𝔻, state_new
end
