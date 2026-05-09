# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

"""
J₂ plasticity with **linear isotropic hardening** only (no kinematic backstress).

Yield surface expands as ``σ_y(\\kappa) = σ_{y0} + H \\kappa`` with accumulated plastic
multiplier `κ` (same convention as [`PlasticStrain`](@ref) / [`EquivalentPlasticStrain`](@ref)
slot `:κ`).

Compare [`PerfectPlasticity`](@ref): fixed yield radius with kinematic translation (`α`).

With ``\\mathrm{seq} = \\sqrt{3/2}\\|\\mathrm{dev}\\,\\sigma\\|`` (von Mises measure used elsewhere in JuliaFEM),
plastic consistency for ``\\boldsymbol{\\varepsilon}_p \\leftarrow \\boldsymbol{\\varepsilon}_p + \\Delta\\lambda\\,\\mathbf{n}``,
``\\mathbf{n}=\\mathrm{dev}\\,\\sigma^\\mathrm{trial}/\\|\\mathrm{dev}\\,\\sigma^\\mathrm{trial}\\|``, and
``\\sigma_y(\\kappa)=\\sigma_{y0}+H\\kappa`` gives
``\\Delta\\lambda = f/(\\sqrt{6}\\,\\mu + H)`` where ``f = \\mathrm{seq}^\\mathrm{trial}-\\sigma_y(\\kappa^{\\mathrm{old}})``.

The algorithmic tangent matches radial-return J₂ with plastic modulus ``\\sqrt{6}\\,\\mu + H``:
``\\mathbb{D}^\\mathrm{alg} = \\mathbb{D}^e - \\dfrac{4\\mu^2}{\\sqrt{6}\\,\\mu + H}\\, \\mathbf{n}\\otimes\\mathbf{n}``.
"""

using Tensors

struct J2LinearIsotropicPlasticity <: AbstractPlasticMaterial
    E::Float64
    ν::Float64
    σ_y0::Float64
    H_iso::Float64
    μ::Float64
    λ::Float64

    function J2LinearIsotropicPlasticity(E::Float64, ν::Float64, σ_y0::Float64, H_iso::Float64)
        E > 0.0 || throw(ArgumentError("Young's modulus must be positive"))
        -1.0 < ν < 0.5 || throw(ArgumentError("Poisson's ratio out of range"))
        σ_y0 > 0.0 || throw(ArgumentError("initial yield σ_y0 must be positive"))
        H_iso ≥ 0.0 || throw(ArgumentError("isotropic hardening modulus must be non-negative"))
        μ = E / (2(1 + ν))
        λ = E * ν / ((1 + ν) * (1 - 2ν))
        new(E, ν, σ_y0, H_iso, μ, λ)
    end
end

function J2LinearIsotropicPlasticity(; E::Real, ν::Real, σ_y0::Real, H_iso::Real)
    J2LinearIsotropicPlasticity(Float64(E), Float64(ν), Float64(σ_y0), Float64(H_iso))
end

material_behavior(::J2LinearIsotropicPlasticity) = StatefulStrainDependent()
supported_physics(::J2LinearIsotropicPlasticity) = (Elasticity{3}(),)
required_state_variables(::J2LinearIsotropicPlasticity) = (PlasticStrain, EquivalentPlasticStrain)

function compute_stress(
    m::J2LinearIsotropicPlasticity,
    ε::SymmetricTensor{2,3},
    ::Nothing,
    Δt::Float64,
)
    return compute_stress(m, ε, NamedTuple(), Δt)
end

function compute_stress(
    m::J2LinearIsotropicPlasticity,
    ε::SymmetricTensor{2,3},
    state_old::NamedTuple,
    Δt::Float64,
)
    μ = m.μ
    λ = m.λ
    H = m.H_iso

    ε_p_old = get(state_old, :ε_p, zero(SymmetricTensor{2,3}))
    κ_old = get(state_old, :κ, 0.0)
    σ_y_trial = m.σ_y0 + H * κ_old

    ε_e = ε - ε_p_old
    I = one(ε)
    σ_trial = λ * tr(ε_e) * I + 2μ * ε_e
    s_trial = dev(σ_trial)

    s_norm_sq = s_trial ⊡ s_trial
    s_norm = √(s_norm_sq)
    seq_trial = √(3.0 / 2.0) * s_norm
    f_trial = seq_trial - σ_y_trial

    if f_trial ≤ 0.0 || s_norm ≤ 1e-30
        𝔻_e = λ * I ⊗ I + 2μ * symmetric_identity_tensor()
        return σ_trial, 𝔻_e, (ε_p = ε_p_old, κ = κ_old)
    end

    n = s_trial / s_norm
    denom = √6 * μ + H
    Δλ = f_trial / denom

    σ = σ_trial - 2μ * Δλ * n
    ε_p_new = ε_p_old + Δλ * n
    κ_new = κ_old + Δλ

    𝔻_e = λ * I ⊗ I + 2μ * symmetric_identity_tensor()
    𝔻 = 𝔻_e - (4μ^2 / denom) * (n ⊗ n)

    return σ, 𝔻, (ε_p = ε_p_new, κ = κ_new)
end
