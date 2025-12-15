"""
Finite Strain Plasticity with Multiplicative Decomposition

Implements J2 plasticity in the finite deformation regime using multiplicative decomposition F = F^e · F^p.
"""

using Tensors
using LinearAlgebra

"""
    FiniteStrainPlasticityState

State variables for finite strain plasticity.

# Fields
- `F_p::Tensor{2,3,Float64,9}` - Plastic deformation gradient
- `α_bar::SymmetricTensor{2,3,Float64,6}` - Backstress
- `κ::Float64` - Equivalent plastic strain (≥ 0)
"""
struct FiniteStrainPlasticityState
    F_p::Tensor{2,3,Float64,9}
    α_bar::SymmetricTensor{2,3,Float64,6}
    κ::Float64

    function FiniteStrainPlasticityState(
        F_p::Tensor{2,3,Float64,9}=one(Tensor{2,3,Float64}),
        α_bar::SymmetricTensor{2,3,Float64,6}=zero(SymmetricTensor{2,3,Float64,6}),
        κ::Float64=0.0
    )
        κ < 0.0 && throw(ArgumentError("κ must be non-negative, got $κ"))
        abs(det(F_p) - 1.0) > 1e-10 && @warn "det(F_p) = $(det(F_p)) ≠ 1 (plastic incompressibility violation)"
        new(F_p, α_bar, κ)
    end
end

"""
    FiniteStrainPlasticity <: AbstractPlasticMaterial

J2 plasticity with finite deformations using multiplicative decomposition.

# Fields
- `E::Float64` - Young's modulus [Pa]
- `ν::Float64` - Poisson's ratio [-]
- `σ_y::Float64` - Yield stress [Pa]
- `H::Float64` - Hardening modulus [Pa]
"""
struct FiniteStrainPlasticity <: AbstractPlasticMaterial
    E::Float64
    ν::Float64
    σ_y::Float64
    H::Float64
    μ::Float64
    λ::Float64

    function FiniteStrainPlasticity(; E::Float64, ν::Float64, σ_y::Float64, H::Float64)
        E <= 0.0 && throw(ArgumentError("Young's modulus E must be positive, got $E"))
        ν <= 0.0 && throw(ArgumentError("Poisson's ratio ν must be positive, got $ν"))
        ν >= 0.5 && throw(ArgumentError("Poisson's ratio ν must be < 0.5 (compressibility), got $ν"))
        σ_y <= 0.0 && throw(ArgumentError("Yield stress σ_y must be positive, got $σ_y"))
        H < 0.0 && throw(ArgumentError("Hardening modulus H must be non-negative, got $H"))

        μ = E / (2 * (1 + ν))
        λ = E * ν / ((1 + ν) * (1 - 2ν))

        new(E, ν, σ_y, H, μ, λ)
    end
end

"""
    compute_stress(material::FiniteStrainPlasticity, F, state_old, Δt) -> (σ, 𝔸, state_new)

Compute Cauchy stress, spatial tangent, and updated state for finite strain plasticity.

Uses multiplicative decomposition F = F^e · F^p with radial return mapping.
"""
function compute_stress(
    material::FiniteStrainPlasticity,
    F::Tensor{2,3},
    state_old::Union{Nothing,FiniteStrainPlasticityState}=nothing,
    Δt::Float64=0.0
)
    # Extract material parameters
    μ = material.μ
    λ = material.λ
    σ_y = material.σ_y
    H = material.H

    # Initialize state if needed
    if state_old === nothing
        state_old = FiniteStrainPlasticityState()
    end

    # Extract old state
    F_p_old = state_old.F_p
    α_bar_old = state_old.α_bar
    κ_old = state_old.κ

    # ====================
    # STEP 1: ELASTIC TRIAL
    # ====================
    # Compute elastic trial: F_e^trial = F · inv(F_p^old)
    F_e_trial = F ⋅ inv(F_p_old)

    # Right Cauchy-Green tensor: C_e^trial = F_e^T · F_e
    C_e_trial = transpose(F_e_trial) ⋅ F_e_trial

    # Elastic volume change
    J_e = det(F_e_trial)

    # Modified elastic deformation (Neo-Hookean)
    C_e_bar = (J_e^(-2 / 3)) * C_e_trial
    I_C = tr(C_e_bar)

    # Mandel stress (work conjugate to C_e)
    # τ = ∂ψ/∂E_e = C_e : S where S is 2nd PK stress
    # For Neo-Hookean: τ = μ·dev(b_e_bar) + K·(J_e - 1)·I
    # In intermediate config: τ = μ·(C_e_bar - I_C/3·I) + λ·ln(J_e)·C_e

    I = one(C_e_trial)

    # Kirchhoff stress (spatial form of Mandel stress)
    # τ_trial = μ·(C_e_bar - I_C/3·I) + λ·ln(J_e)·C_e_trial
    τ_trial = μ * (C_e_bar - (I_C / 3) * I) + λ * log(J_e) * C_e_trial

    # Make symmetric (should be symmetric already, but numerical precision)
    τ_trial = symmetric(τ_trial)

    # ====================
    # STEP 2: YIELD CHECK
    # ====================
    # Relative Mandel stress (shifted by backstress)
    s_trial = dev(τ_trial - α_bar_old)

    # von Mises equivalent stress in intermediate config
    # Note: Different normalization than small strain!
    # Here: f = ||s|| - √(2/3) σ_y
    s_trial_norm = √(s_trial ⊡ s_trial)
    f_trial = s_trial_norm - √(2 / 3) * σ_y

    # ====================
    # STEP 3: RETURN MAPPING
    # ====================
    if f_trial ≤ 0.0
        # ==================== ELASTIC ====================
        F_p_new = F_p_old
        α_bar_new = α_bar_old
        κ_new = κ_old
        τ = τ_trial

        # Elastic tangent (push-forward to spatial config below)

    else
        # ==================== PLASTIC ====================
        # Flow direction (unit tensor)
        n = s_trial / s_trial_norm

        # Plastic multiplier (similar to small strain but with √(2/3) normalization)
        # Derivation: ||s_trial - (2μ + 2H/3)Δγ·n|| = √(2/3)σ_y
        # Δγ = (s_trial_norm - √(2/3)σ_y) / (2μ + 2H/3)
        Δγ = f_trial / (2μ + (2.0 / 3.0) * H)

        # Update Mandel stress (radial return)
        τ = τ_trial - 2μ * Δγ * n

        # Update backstress (kinematic hardening in intermediate config)
        α_bar_new = α_bar_old + (2.0 / 3.0) * H * Δγ * n

        # Update equivalent plastic strain
        κ_new = κ_old + √(2 / 3) * Δγ

        # Update plastic deformation gradient using exponential map
        # F_p_new = exp(Δγ · n) · F_p_old
        # For small Δγ: exp(Δγ·n) ≈ I + Δγ·n (first-order approximation)
        # For general case: use exponential map (more complex)

        # Simplified: First-order exponential map
        # This is valid for small plastic increments (Δγ << 1)
        # For large increments, would need full exponential map
        exp_map = I + Δγ * n
        F_p_new = exp_map ⋅ F_p_old

        # Note: This can violate det(F_p) = 1 for large steps
        # In production code, would need to project onto SL(3) or use better integrator
    end

    # ====================
    # STEP 4: PUSH-FORWARD TO SPATIAL CONFIGURATION
    # ====================
    # Cauchy stress: σ = (1/J) F_e · τ · F_e^T
    # Since τ is in intermediate config, need to push forward

    # Current elastic deformation
    F_e = F ⋅ inv(F_p_new)
    J = det(F)

    # Push-forward Kirchhoff stress to spatial config
    # τ_spatial = F_e · τ · F_e^T
    τ_spatial = F_e ⋅ τ ⋅ transpose(F_e)

    # Cauchy stress
    σ = (1.0 / J) * symmetric(τ_spatial)

    # ====================
    # STEP 5: CONSISTENT TANGENT
    # ====================
    # Spatial tangent: 𝔸 = ∂σ/∂F
    # For finite strain, this is extremely complex
    # Simplified: Use elastic tangent (loses quadratic convergence but simpler)

    # Elastic tangent in intermediate config
    𝔻_e = λ * (I ⊗ I) + 2μ * symmetric_identity_tensor()

    # Push-forward to spatial config (simplified)
    # Full derivation requires tensor transformation rules
    # For now: use elastic tangent as approximation
    𝔸 = 𝔻_e  # This is NOT correct for finite strain! Placeholder.

    # TODO: Implement proper spatial tangent for finite strain
    # Requires: ∂σ/∂F = f(F, F_p, τ, 𝔻^ep)
    # See Simo & Hughes Box 9.4 for full algorithm

    # New state
    state_new = FiniteStrainPlasticityState(F_p_new, α_bar_new, κ_new)

    return σ, 𝔸, state_new
end

"""
    symmetric_identity_tensor()

Fourth-order symmetric identity tensor: 𝕀 = ½(δᵢₖδⱼₗ + δᵢₗδⱼₖ)
"""
@inline function symmetric_identity_tensor()
    return SymmetricTensor{4,3}((i, j, k, l) ->
        (i == k && j == l ? 0.5 : 0.0) + (i == l && j == k ? 0.5 : 0.0))
end
