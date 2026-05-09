# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

"""
Additional compressible hyperelastic potentials using Green–Lagrange strain `E`
through `C = 2E + I` and automatic differentiation (`Tensors.hessian`), matching
[`NeoHookean`](@ref).

Models: [`MooneyRivlin`](@ref), [`Yeoh3`](@ref), [`Gent`](@ref).
"""

using Tensors

@inline function _hyperelastic_J(C::SymmetricTensor{2,3})
    d = det(C)
    d > 0 || throw(DomainError(d, "det(C) must be positive"))
    return √d
end

@inline function _second_invariant_C(C::SymmetricTensor{2,3})
    trC = tr(C)
    trCC = tr(C ⋅ C)
    return 0.5 * (trC * trC - trCC)
end

function _hyperelastic_stress_tangent(ψ, C::SymmetricTensor{2,3})
    ∂²ψ∂C², ∂ψ∂C = Tensors.hessian(ψ, C, :all)
    S = 2 * ∂ψ∂C
    𝔻 = 4 * ∂²ψ∂C²
    return S, 𝔻
end

function _compute_from_E(material, E::SymmetricTensor{2,3,T}) where T
    I = one(E)
    C = 2E + I
    ψ(C_) = strain_energy(material, C_)
    S, 𝔻 = _hyperelastic_stress_tangent(ψ, C)
    return S, 𝔻, NamedTuple()
end

"""
    MooneyRivlin <: AbstractElasticMaterial

Compressible Mooney–Rivlin strain energy:

`ψ = C₁₀ (I₁ − 3) + C₀₁ (I₂ − 3) + (κ/2)(J − 1)²`

with `I₁ = tr(C)`, `I₂` the standard second invariant, `J = √det(C)`.
"""
struct MooneyRivlin <: AbstractElasticMaterial
    C10::Float64
    C01::Float64
    κ_bulk::Float64

    function MooneyRivlin(C10::Float64, C01::Float64, κ_bulk::Float64)
        κ_bulk > 0 || throw(ArgumentError("bulk modulus κ_bulk must be positive"))
        new(C10, C01, κ_bulk)
    end
end

MooneyRivlin(; C10::Real, C01::Real, κ_bulk::Real) =
    MooneyRivlin(Float64(C10), Float64(C01), Float64(κ_bulk))

material_behavior(::MooneyRivlin) = StatelessStrainDependent()
supported_physics(::MooneyRivlin) = (Elasticity{3}(),)
required_state_variables(::MooneyRivlin) = ()

function strain_energy(m::MooneyRivlin, C::SymmetricTensor{2,3})
    J = _hyperelastic_J(C)
    I1 = tr(C)
    I2 = _second_invariant_C(C)
    return m.C10 * (I1 - 3) + m.C01 * (I2 - 3) + (m.κ_bulk / 2) * (J - 1)^2
end

function compute_stress(m::MooneyRivlin, E::SymmetricTensor{2,3,T}, ::Nothing, Δt::Float64) where {T}
    return _compute_from_E(m, E)
end

function compute_stress(m::MooneyRivlin, E::SymmetricTensor{2,3,T}, ::NamedTuple, Δt::Float64) where {T}
    return _compute_from_E(m, E)
end

compute_stress(m::MooneyRivlin, E::SymmetricTensor{2,3}) = compute_stress(m, E, nothing, 0.0)

"""
    Yeoh3 <: AbstractElasticMaterial

Three-term Yeoh expansion in `I₁`:

`ψ = Σᵢ Cᵢ₀ (I₁ − 3)ⁱ` for `i ∈ {1,2,3}`, plus `(κ/2)(J − 1)²`.
"""
struct Yeoh3 <: AbstractElasticMaterial
    C10::Float64
    C20::Float64
    C30::Float64
    κ_bulk::Float64

    function Yeoh3(C10::Float64, C20::Float64, C30::Float64, κ_bulk::Float64)
        κ_bulk > 0 || throw(ArgumentError("bulk modulus κ_bulk must be positive"))
        new(C10, C20, C30, κ_bulk)
    end
end

Yeoh3(; C10::Real, C20::Real = 0.0, C30::Real = 0.0, κ_bulk::Real) =
    Yeoh3(Float64(C10), Float64(C20), Float64(C30), Float64(κ_bulk))

material_behavior(::Yeoh3) = StatelessStrainDependent()
supported_physics(::Yeoh3) = (Elasticity{3}(),)
required_state_variables(::Yeoh3) = ()

function strain_energy(m::Yeoh3, C::SymmetricTensor{2,3})
    J = _hyperelastic_J(C)
    I1 = tr(C)
    x = I1 - 3
    return m.C10 * x + m.C20 * x^2 + m.C30 * x^3 + (m.κ_bulk / 2) * (J - 1)^2
end

function compute_stress(m::Yeoh3, E::SymmetricTensor{2,3,T}, ::Nothing, Δt::Float64) where {T}
    return _compute_from_E(m, E)
end

function compute_stress(m::Yeoh3, E::SymmetricTensor{2,3,T}, ::NamedTuple, Δt::Float64) where {T}
    return _compute_from_E(m, E)
end

compute_stress(m::Yeoh3, E::SymmetricTensor{2,3}) = compute_stress(m, E, nothing, 0.0)

"""
    Gent <: AbstractElasticMaterial

Gent shear resistance with compressible volumetric penalty:

`ψ = −(μ J_m / 2) log(1 − (I₁ − 3)/J_m) + (κ/2)(J − 1)²`

Requires `I₁ − 3 < J_m`.
"""
struct Gent <: AbstractElasticMaterial
    μ::Float64
    Jm::Float64
    κ_bulk::Float64

    function Gent(μ::Float64, Jm::Float64, κ_bulk::Float64)
        μ > 0 || throw(ArgumentError("μ must be positive"))
        Jm > 0 || throw(ArgumentError("Jm must be positive"))
        κ_bulk > 0 || throw(ArgumentError("bulk modulus κ_bulk must be positive"))
        new(μ, Jm, κ_bulk)
    end
end

Gent(; μ::Real, Jm::Real, κ_bulk::Real) = Gent(Float64(μ), Float64(Jm), Float64(κ_bulk))

material_behavior(::Gent) = StatelessStrainDependent()
supported_physics(::Gent) = (Elasticity{3}(),)
required_state_variables(::Gent) = ()

function strain_energy(m::Gent, C::SymmetricTensor{2,3})
    J = _hyperelastic_J(C)
    I1 = tr(C)
    x = I1 - 3
    x < m.Jm || throw(DomainError(x, "I₁ − 3 must be < Jm for Gent model"))
    return -m.μ * m.Jm / 2 * log(1 - x / m.Jm) + (m.κ_bulk / 2) * (J - 1)^2
end

function compute_stress(m::Gent, E::SymmetricTensor{2,3,T}, ::Nothing, Δt::Float64) where {T}
    return _compute_from_E(m, E)
end

function compute_stress(m::Gent, E::SymmetricTensor{2,3,T}, ::NamedTuple, Δt::Float64) where {T}
    return _compute_from_E(m, E)
end

compute_stress(m::Gent, E::SymmetricTensor{2,3}) = compute_stress(m, E, nothing, 0.0)
