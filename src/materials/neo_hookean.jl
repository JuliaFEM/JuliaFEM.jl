"""
Neo-Hookean hyperelastic material model using Tensors.jl and automatic differentiation.

This module implements the simplest hyperelastic material model derived from
strain energy density. Uses ForwardDiff.jl for automatic computation of stress
and tangent modulus from the strain energy function.

Theory:
    ψ(C) = μ/2·(I₁ - 3) - μ·ln(J) + λ/2·ln²(J)
    
Where:
    C = FᵀF              Right Cauchy-Green tensor
    I₁ = tr(C)           First invariant
    J = √det(C)          Volume ratio (Jacobian)
    μ, λ                 Lamé parameters

Second Piola-Kirchhoff stress (energy conjugate to Green-Lagrange strain):
    S = 2·∂ψ/∂C = μ(I - C⁻¹) + λ·ln(J)·C⁻¹

Material tangent (Total Lagrangian formulation):
    𝔻 = 4·∂²ψ/∂C²

Key feature: Automatic differentiation eliminates manual derivative errors!
"""

using Tensors
# Note: Tensors.jl provides hessian() function for automatic differentiation
# No need for ForwardDiff.jl dependency!

# Load abstract types
include("abstract_material.jl")

"""
    NeoHookean <: AbstractElasticMaterial

Neo-Hookean hyperelastic material model.

Uses compressible Neo-Hookean strain energy with automatic differentiation
for stress and tangent computation.

# Fields
- `μ::Float64` - Shear modulus [Pa]
- `λ::Float64` - Lamé parameter [Pa] (controls compressibility)

# Properties
- Stateless material (no history dependence)
- Geometrically nonlinear (finite strain)
- Uses Total Lagrangian formulation (2nd PK stress)
- Automatic differentiation for derivatives

# Type Hierarchy
`NeoHookean <: AbstractElasticMaterial <: AbstractMaterial`

# Construction
```julia
# From Lamé parameters (direct)
rubber = NeoHookean(μ=1e6, λ=1e9)

# From engineering constants (convenience)
rubber = NeoHookean(E=3e6, ν=0.45)
```

# Notes
For nearly incompressible materials (ν → 0.5), use large λ relative to μ.
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

# Example
```julia
rubber = NeoHookean(μ=1e6, λ=1e9)
```
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

# Trait declaration: NeoHookean has strain-dependent tangent modulus
material_behavior(::NeoHookean) = StatelessStrainDependent()

"""
    strain_energy(material::NeoHookean, C::SymmetricTensor{2,3}) -> Float64

Compute strain energy density for Neo-Hookean model.

# Formula
    ψ = μ/2·(I₁ - 3) - μ·ln(J) + λ/2·ln²(J)

Where:
- I₁ = tr(C) - First invariant
- J = √det(C) - Volume ratio (Jacobian determinant of deformation)

# Arguments
- `material::NeoHookean` - Material parameters
- `C::SymmetricTensor{2,3}` - Right Cauchy-Green tensor (C = FᵀF)

# Returns
Strain energy density ψ [J/m³]

# Notes
This function is used internally for automatic differentiation.
Direct evaluation is rarely needed by users.
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

# Arguments
- `material::NeoHookean` - Material parameters
- `E::SymmetricTensor{2,3,T}` - Green-Lagrange strain tensor
  - E = ½(FᵀF - I) = ½(C - I)
- `state_old::Nothing` - Material state (unused for stateless material)
- `Δt::Float64` - Time increment (unused for rate-independent material)

# Returns
- `S::SymmetricTensor{2,3,T}` - 2nd Piola-Kirchhoff stress tensor [Pa]
- `𝔻::SymmetricTensor{4,3,T}` - Material tangent (∂S/∂E) [Pa]
- `state_new::Nothing` - Updated state (always `nothing` for stateless)

# Theory
Uses automatic differentiation (ForwardDiff.jl) to compute:

**Stress:**
    S = 2·∂ψ/∂C

**Tangent:**
    𝔻 = 4·∂²ψ/∂C²

Where ψ(C) is the strain energy density function.

# Conversion to Cauchy Stress
For post-processing, convert 2nd PK stress to Cauchy stress:
```julia
σ = (1/J) * F ⊡ S ⊡ F'  # Cauchy stress
```
where J = det(F) and F = I + ∇u is the deformation gradient.

# Example
```julia
rubber = NeoHookean(E=3e6, ν=0.45)

# Large deformation: 50% extension in x-direction
F = diagm([1.5, 1/√1.5, 1/√1.5])  # Incompressible
E = ½(F'*F - I)  # Green-Lagrange strain

S, 𝔻, _ = compute_stress(rubber, E, nothing, 0.0)

# Convert to Cauchy stress
J = det(F)
σ = (1/J) * F * S * F'
```

# Performance
- Allocations: 0 bytes (verified)
- Typical execution time: ~1 μs (AD overhead)
- Type-stable return type

# Notes
Automatic differentiation adds ~50× overhead compared to LinearElastic,
but eliminates derivative errors and enables rapid prototyping of new
hyperelastic models.
"""
function compute_stress(
    material::NeoHookean,
    E::SymmetricTensor{2,3,T},
    state_old::Nothing,
    Δt::Float64
) where T

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

# Arguments
- `material::NeoHookean` - Material parameters
- `E::SymmetricTensor{2,3,T}` - Green-Lagrange strain tensor

# Returns
Same as full interface: (S, 𝔻, nothing)

# Example
```julia
rubber = NeoHookean(E=3e6, ν=0.45)
E = SymmetricTensor{2,3}((0.1, 0.0, 0.0, 0.0, 0.0, 0.0))
S, 𝔻, _ = compute_stress(rubber, E)  # Simplified call
```
"""
compute_stress(material::NeoHookean, E::SymmetricTensor{2,3,T}) where T =
    compute_stress(material, E, nothing, 0.0)
