# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

"""
J₂ plasticity with a **St. Venant–Kirchhoff** elastic relation written on the
Green–Lagrange strain `E` from the assembly (`update_material_cache!` uses
[`GreenLagrangeKinematics`](@ref)).

This is **not** multiplicative finite-strain plasticity; it is the common
total-Lagrangian approximation that keeps the same radial return as
[`PerfectPlasticity`](@ref) but replaces small-strain `ε` with `E`.

See [`continuum_kinematics`](@ref).
"""

using Tensors

struct StVenantKirchhoffJ2Plasticity <: AbstractPlasticMaterial
    E::Float64
    ν::Float64
    σ_y::Float64
    H::Float64
    μ::Float64
    λ::Float64

    function StVenantKirchhoffJ2Plasticity(E::Float64, ν::Float64, σ_y::Float64, H::Float64)
        E > 0.0 || throw(ArgumentError("Young's modulus must be positive"))
        -1.0 < ν < 0.5 || throw(ArgumentError("Poisson's ratio out of range"))
        σ_y > 0.0 || throw(ArgumentError("Yield stress must be positive"))
        H ≥ 0.0 || throw(ArgumentError("Hardening modulus must be non-negative"))
        μ = E / (2(1 + ν))
        λ = E * ν / ((1 + ν) * (1 - 2ν))
        new(E, ν, σ_y, H, μ, λ)
    end
end

StVenantKirchhoffJ2Plasticity(; E::Real, ν::Real, σ_y::Real, H::Real) =
    StVenantKirchhoffJ2Plasticity(Float64(E), Float64(ν), Float64(σ_y), Float64(H))

material_behavior(::StVenantKirchhoffJ2Plasticity) = StatefulStrainDependent()
supported_physics(::StVenantKirchhoffJ2Plasticity) = (Elasticity{3}(),)
required_state_variables(::StVenantKirchhoffJ2Plasticity) =
    (PlasticStrain, Backstress, EquivalentPlasticStrain)

continuum_kinematics(::StVenantKirchhoffJ2Plasticity) = GreenLagrangeKinematics()

function compute_stress(
    m::StVenantKirchhoffJ2Plasticity,
    Estrain::SymmetricTensor{2,3},
    ::Nothing,
    Δt::Float64,
)
    return compute_stress(m, Estrain, NamedTuple(), Δt)
end

function compute_stress(
    m::StVenantKirchhoffJ2Plasticity,
    Estrain::SymmetricTensor{2,3},
    state_old::NamedTuple,
    Δt::Float64,
)
    μ = m.μ
    λ = m.λ
    σ_y = m.σ_y
    H = m.H

    ε_p_old = get(state_old, :ε_p, zero(SymmetricTensor{2,3}))
    α_old = get(state_old, :α, zero(SymmetricTensor{2,3}))
    κ_old = get(state_old, :κ, 0.0)

    ε_e = Estrain - ε_p_old
    I = one(Estrain)
    σ_trial = λ * tr(ε_e) * I + 2μ * ε_e

    s_trial = dev(σ_trial - α_old)
    s_trial_norm = √(3 / 2) * √(s_trial ⊡ s_trial)
    f_trial = s_trial_norm - σ_y

    if f_trial ≤ 0.0
        σ = σ_trial
        state_new = (ε_p=ε_p_old, α=α_old, κ=κ_old)
        𝔻 = λ * I ⊗ I + 2μ * symmetric_identity_tensor()
    else
        n = s_trial / s_trial_norm
        Δλ = f_trial / (2μ + (2.0 / 3.0) * H)
        σ = σ_trial - 2μ * Δλ * n
        α_new = α_old + (2.0 / 3.0) * H * Δλ * n
        ε_p_new = ε_p_old + Δλ * n
        κ_new = κ_old + Δλ
        state_new = (ε_p=ε_p_new, α=α_new, κ=κ_new)
        𝔻_e = λ * I ⊗ I + 2μ * symmetric_identity_tensor()
        𝔻 = 𝔻_e - (4μ^2 / (2μ + (2.0 / 3.0) * H)) * (n ⊗ n)
    end

    return σ, 𝔻, state_new
end
