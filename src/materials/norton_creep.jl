# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

"""
Norton deviatoric creep strain `ε_c` with linear elastic unloading:

`σ = 𝔻 : (ε − ε_c)`, `Δε_c = (3/2) Δt A σ_vm^{n−1} s / σ_vm`.

Optional irradiation-like **volumetric swelling** increment
`Δε_sw = β \\, \\dot{\\phi} \\, Δt` added isotropically to `ε_c` when `β` and
`phi_dot` are set.

Elastic tangent only (ignores creep Jacobian); suited to explicit creep
sub-stepping inside an implicit displacement solve.
"""

using Tensors

struct NortonCreepElastic <: AbstractMaterial
    elastic::LinearElastic
    A::Float64
    n::Float64
    β_swelling::Float64
    phi_dot::Float64

    function NortonCreepElastic(elastic::LinearElastic, A::Float64, n::Float64, β::Float64, ϕdot::Float64)
        A ≥ 0 || throw(ArgumentError("Norton coefficient A must be non-negative"))
        β ≥ 0 || throw(ArgumentError("swelling coefficient β must be non-negative"))
        ϕdot ≥ 0 || throw(ArgumentError("phi_dot must be non-negative"))
        new(elastic, A, n, β, ϕdot)
    end
end

function NortonCreepElastic(; E::Real, ν::Real, A::Real, n::Real, β_swelling::Real = 0.0, phi_dot::Real = 0.0)
    NortonCreepElastic(LinearElastic(E = E, ν = ν), Float64(A), Float64(n), Float64(β_swelling), Float64(phi_dot))
end

material_behavior(::NortonCreepElastic) = StatefulStrainDependent()
supported_physics(::NortonCreepElastic) = (Elasticity{3}(),)
required_state_variables(::NortonCreepElastic) = (CreepStrain,)

function compute_stress(mat::NortonCreepElastic, ε::SymmetricTensor{2,3}, ::Nothing, Δt::Float64)
    return compute_stress(mat, ε, NamedTuple(), Δt)
end

function compute_stress(mat::NortonCreepElastic, ε::SymmetricTensor{2,3}, state_old::NamedTuple, Δt::Float64)
    ε_c_old = get(state_old, :ε_c, zero(SymmetricTensor{2,3}))

    σ_el, _, _ = compute_stress(mat.elastic, ε - ε_c_old, NamedTuple(), 0.0)
    s = dev(σ_el)
    norm2 = s ⊡ s
    seq = √(3.0 / 2.0 * norm2)

    Δε_vol = zero(SymmetricTensor{2,3})
    if mat.β_swelling > 0 && mat.phi_dot > 0 && Δt > 0
        ev = mat.β_swelling * mat.phi_dot * Δt
        Δε_vol = (ev / 3.0) * one(SymmetricTensor{2,3,Float64,6})
    end

    Δε_creep = zero(SymmetricTensor{2,3})
    if mat.A > 0 && Δt > 0 && seq > 1e-14
        η = Δt * mat.A * seq^(mat.n - 1)
        Δε_creep = (1.5 * η / seq) * s
    end

    ε_c_new = ε_c_old + Δε_creep + Δε_vol
    σ_new, _, _ = compute_stress(mat.elastic, ε - ε_c_new, NamedTuple(), 0.0)
    𝔻 = elasticity_tensor(mat.elastic)

    return σ_new, 𝔻, (ε_c=ε_c_new,)
end
