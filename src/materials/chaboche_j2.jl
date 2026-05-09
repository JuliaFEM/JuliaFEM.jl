# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

"""
Two-surface Armstrong–Frederick–style kinematic hardening with **explicit**
backstress relaxation:

`αᵢ ← αᵢ + (2/3) Cᵢ Δλ n − γᵢ Δλ αᵢ`

Yield surface uses `η = dev(σ_trial − α₁ − α₂)`. Plastic modulus in the radial
return uses `H_eff = C₁ + C₂` in the same slot as [`PerfectPlasticity`](@ref)`s
`H`. This is a pragmatic explicit operator-split suitable for small strain steps;
fully implicit consistent tangents are not implemented.
"""

using Tensors

struct ChabocheJ2Plasticity <: AbstractPlasticMaterial
    E::Float64
    ν::Float64
    σ_y::Float64
    C1::Float64
    γ1::Float64
    C2::Float64
    γ2::Float64
    μ::Float64
    λ::Float64

    function ChabocheJ2Plasticity(
            E::Float64, ν::Float64, σ_y::Float64,
            C1::Float64, γ1::Float64, C2::Float64, γ2::Float64,
        )
        E > 0 || throw(ArgumentError("E must be positive"))
        -1 < ν < 0.5 || throw(ArgumentError("ν out of range"))
        σ_y > 0 || throw(ArgumentError("σ_y must be positive"))
        C1 ≥ 0 && C2 ≥ 0 || throw(ArgumentError("Ci must be non-negative"))
        γ1 ≥ 0 && γ2 ≥ 0 || throw(ArgumentError("γi must be non-negative"))
        μ = E / (2(1 + ν))
        λ = E * ν / ((1 + ν) * (1 - 2ν))
        new(E, ν, σ_y, C1, γ1, C2, γ2, μ, λ)
    end
end

function ChabocheJ2Plasticity(;
        E::Real, ν::Real, σ_y::Real,
        C1::Real, γ1::Real, C2::Real, γ2::Real,
    )
    ChabocheJ2Plasticity(Float64(E), Float64(ν), Float64(σ_y),
        Float64(C1), Float64(γ1), Float64(C2), Float64(γ2))
end

material_behavior(::ChabocheJ2Plasticity) = StatefulStrainDependent()
supported_physics(::ChabocheJ2Plasticity) = (Elasticity{3}(),)
required_state_variables(::ChabocheJ2Plasticity) =
    (PlasticStrain, ChabocheAlpha{1}, ChabocheAlpha{2}, EquivalentPlasticStrain)

function compute_stress(m::ChabocheJ2Plasticity, ε::SymmetricTensor{2,3}, ::Nothing, Δt::Float64)
    return compute_stress(m, ε, NamedTuple(), Δt)
end

function compute_stress(m::ChabocheJ2Plasticity, ε::SymmetricTensor{2,3}, state_old::NamedTuple, Δt::Float64)
    μ = m.μ
    λ = m.λ
    σ_y = m.σ_y
    H_eff = m.C1 + m.C2

    ε_p_old = get(state_old, :ε_p, zero(SymmetricTensor{2,3}))
    α1_old = get(state_old, :α1, zero(SymmetricTensor{2,3}))
    α2_old = get(state_old, :α2, zero(SymmetricTensor{2,3}))
    κ_old = get(state_old, :κ, 0.0)

    ε_e = ε - ε_p_old
    I = one(ε)
    σ_trial = λ * tr(ε_e) * I + 2μ * ε_e

    α_tot = α1_old + α2_old
    s_trial = dev(σ_trial - α_tot)
    s_norm = √(s_trial ⊡ s_trial)
    seq = √(3 / 2) * s_norm
    f_trial = seq - σ_y

    if f_trial ≤ 0.0 || s_norm ≤ 1e-30
        σ = σ_trial
        state_new = (ε_p=ε_p_old, α1=α1_old, α2=α2_old, κ=κ_old)
        𝔻 = λ * I ⊗ I + 2μ * symmetric_identity_tensor()
        return σ, 𝔻, state_new
    end

    n = s_trial / s_norm
    Δλ = f_trial / (2μ + (2.0 / 3.0) * H_eff)

    σ = σ_trial - 2μ * Δλ * n
    ε_p_new = ε_p_old + Δλ * n

    α1_new = α1_old + (2.0 / 3.0) * m.C1 * Δλ * n - m.γ1 * Δλ * α1_old
    α2_new = α2_old + (2.0 / 3.0) * m.C2 * Δλ * n - m.γ2 * Δλ * α2_old

    κ_new = κ_old + Δλ
    state_new = (ε_p=ε_p_new, α1=α1_new, α2=α2_new, κ=κ_new)

    𝔻_e = λ * I ⊗ I + 2μ * symmetric_identity_tensor()
    𝔻 = 𝔻_e - (4μ^2 / (2μ + (2.0 / 3.0) * H_eff)) * (n ⊗ n)

    return σ, 𝔻, state_new
end
