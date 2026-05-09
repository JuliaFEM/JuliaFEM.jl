# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

"""
Scalar isotropic damage on a linear elastic baseline (`σ̃ = 𝔻ᵉ : ε`), with

`σ = (1 − d) σ̃`, `d = d_max (1 − exp(−r ⟨κ − ε₀⟩₊))`, `κ = max(κ_old, ε_eq)`,

`ε_eq = √(2/3 ‖dev(ε)‖²)`. The tangent omits `∂d/∂ε` (explicit damage stagger).
"""

using Tensors

struct ScalarDamageLinearElastic <: AbstractPlasticMaterial
    elastic::LinearElastic
    r::Float64
    ε0::Float64
    d_max::Float64

    function ScalarDamageLinearElastic(elastic::LinearElastic, r::Float64, ε0::Float64, d_max::Float64)
        r ≥ 0 || throw(ArgumentError("damage rate r must be non-negative"))
        0 ≤ d_max ≤ 1 || throw(ArgumentError("d_max must lie in [0,1]"))
        new(elastic, r, ε0, d_max)
    end
end

function ScalarDamageLinearElastic(; E::Real, ν::Real, r::Real, ε0::Real = 0.0, d_max::Real = 1.0)
    return ScalarDamageLinearElastic(LinearElastic(E = E, ν = ν), Float64(r), Float64(ε0), Float64(d_max))
end

material_behavior(::ScalarDamageLinearElastic) = StatefulStrainDependent()
supported_physics(::ScalarDamageLinearElastic) = (Elasticity{3}(),)
required_state_variables(::ScalarDamageLinearElastic) = (DamageVariable, DamageEquivalentStrain)

function compute_stress(
    mat::ScalarDamageLinearElastic,
    ε::SymmetricTensor{2,3},
    ::Nothing,
    Δt::Float64,
)
    return compute_stress(mat, ε, NamedTuple(), Δt)
end

function compute_stress(
    mat::ScalarDamageLinearElastic,
    ε::SymmetricTensor{2,3},
    state_old::NamedTuple,
    Δt::Float64,
)
    d_old = get(state_old, :d, 0.0)
    κ_old = get(state_old, :κ_d, 0.0)

    sdev = dev(ε)
    ε_eq = √(2.0 / 3.0 * (sdev ⊡ sdev))
    κ_new = max(κ_old, ε_eq)
    soft = κ_new - mat.ε0
    d_new = soft ≤ 0.0 ? d_old :
            min(mat.d_max, 1 - exp(-mat.r * soft))

    σ_tilde, 𝔻_e, _ = compute_stress(mat.elastic, ε, NamedTuple(), 0.0)
    σ = (1 - d_new) * σ_tilde
    𝔻 = (1 - d_new) * 𝔻_e

    state_new = (d=d_new, κ_d=κ_new)
    return σ, 𝔻, state_new
end
