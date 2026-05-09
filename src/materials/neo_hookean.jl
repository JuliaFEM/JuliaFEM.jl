"""
Neo-Hookean hyperelastic material model using Tensors.jl and automatic differentiation.
"""

using Tensors
# Note: Tensors.jl provides hessian() function for automatic differentiation
# No need for ForwardDiff.jl dependency!

"""
    NeoHookean <: AbstractElasticMaterial

Neo-Hookean hyperelastic material model.

# Fields
- `μ::Float64` - Shear modulus [Pa]
- `λ::Float64` - Lamé parameter [Pa] (controls compressibility)
"""
struct NeoHookean <: AbstractElasticMaterial
    μ::Float64  # Shear modulus [Pa]
    λ::Float64  # Lamé parameter [Pa]

    function NeoHookean(μ::Float64, λ::Float64)
        μ > 0.0 || throw(ArgumentError("Shear modulus μ must be positive, got μ = $μ"))
        λ > 0.0 || throw(ArgumentError("Lamé parameter λ must be positive, got λ = $λ"))
        new(μ, λ)
    end
end

"""
    NeoHookean(; μ, λ)

Convenience constructor with keyword arguments (Lamé parameters).
"""
function NeoHookean(; μ::Real=NaN, λ::Real=NaN, E_mod::Real=NaN, nu::Real=NaN)
    # Check which set of parameters was provided
    if !isnan(μ) && !isnan(λ)
        # Lamé parameters provided
        return NeoHookean(Float64(μ), Float64(λ))
    elseif !isnan(E_mod) && !isnan(nu)
        # Engineering constants provided
        E_mod > 0.0 || throw(ArgumentError("Young's modulus E_mod must be positive, got E_mod = $E_mod"))
        -1.0 < nu < 0.5 || throw(ArgumentError("Poisson's ratio must satisfy -1 < nu < 0.5, got nu = $nu"))

        μ_val = E_mod / (2(1 + nu))
        λ_val = E_mod * nu / ((1 + nu) * (1 - 2nu))

        return NeoHookean(Float64(μ_val), Float64(λ_val))
    else
        throw(ArgumentError("Must provide either (μ, λ) or (E_mod, nu)"))
    end
end

material_behavior(::NeoHookean) = StatelessStrainDependent()
supported_physics(::NeoHookean) = (Elasticity{3}(),)
required_state_variables(::NeoHookean) = ()

"""
    strain_energy(material::NeoHookean, C::SymmetricTensor{2,3}) -> Float64

Compute strain energy density: ψ = μ/2·(I₁ - 3) - μ·ln(J) + λ/2·ln²(J)
"""
function strain_energy(material::NeoHookean, C::SymmetricTensor{2,3})
    μ, λ = material.μ, material.λ

    # Invariants
    I₁ = tr(C)
    J = √(det(C))

    # Guard against invalid deformation (negative Jacobian)
    J > 0.0 || throw(DomainError(J, "Jacobian J = √det(C) must be positive"))

    # Strain energy: ψ = μ/2·(I₁ - 3) - μ·ln(J) + λ/2·ln²(J)
    ψ = μ / 2 * (I₁ - 3) - μ * log(J) + λ / 2 * log(J)^2

    return ψ
end

"""
    compute_stress(material::NeoHookean, E, state_old, Δt) -> (S, 𝔻, state_new)

Compute stress and tangent modulus for Neo-Hookean material using automatic differentiation.

Uses automatic differentiation to compute S = 2·∂ψ/∂C and 𝔻 = 4·∂²ψ/∂C².
"""
function compute_stress(
    material::NeoHookean,
    E::SymmetricTensor{2,3,T},
    state_old::Nothing,
    Δt::Float64,
) where {T}
    return _compute_stress_neo_hookean(material, E)
end

function compute_stress(
    material::NeoHookean,
    E::SymmetricTensor{2,3,T},
    state_old::NamedTuple,
    Δt::Float64,
) where {T}
    return _compute_stress_neo_hookean(material, E)
end

function _compute_stress_neo_hookean(
    material::NeoHookean,
    E::SymmetricTensor{2,3,T},
) where {T}
    # Right Cauchy-Green tensor: C = 2E + I
    I = one(E)
    C = 2E + I

    # Strain energy function (closure capturing material parameters)
    ψ(C_) = strain_energy(material, C_)

    # Automatic differentiation!
    # gradient:  ∂ψ/∂C
    # hessian: ∂²ψ/∂C²
    ∂²ψ∂C², ∂ψ∂C = Tensors.hessian(ψ, C, :all)

    # Second Piola-Kirchhoff stress: S = 2·∂ψ/∂C
    S = 2 * ∂ψ∂C

    # Material tangent: 𝔻 = 4·∂²ψ/∂C²
    𝔻 = 4 * ∂²ψ∂C²

    return S, 𝔻, nothing  # No state change (stateless material)
end

"""
    compute_stress(material::NeoHookean, E::SymmetricTensor{2,3,T}) -> (S, 𝔻, nothing)

Simplified interface without state management for stateless material.
"""
compute_stress(material::NeoHookean, E::SymmetricTensor{2,3,T}) where {T} =
    compute_stress(material, E, nothing, 0.0)
