# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Material model API definitions.

This file defines material-specific abstract types and interfaces.
Must be included after core api.jl.
"""

# ============================================================================
# MATERIAL MODEL ABSTRACTIONS
# ============================================================================

"""
    AbstractMaterial

Abstract type for all material models.

# Interface Requirements

All material models must implement:
```julia
compute_stress(
    material::AbstractMaterial,
    ε::SymmetricTensor{2,3,T},
    state_old,
    Δt::Float64
) -> (σ::SymmetricTensor{2,3,T}, 𝔻::SymmetricTensor{4,3,T}, state_new)
```

Returns:
- `σ`: Cauchy stress tensor
- `𝔻`: Material tangent (4th-order elasticity tensor)
- `state_new`: Updated material state (for plasticity, damage, etc.)

# Type Hierarchy
- `AbstractElasticMaterial` - Stateless elastic materials (no history)
- `AbstractPlasticMaterial` - Stateful plastic materials (history-dependent)

# Design Philosophy

Materials are immutable structs with parameters (E, ν, etc.).
Material state (plastic strain, damage) stored separately in solution.
Use Tensors.jl for all tensor operations (no Voigt notation).

# See Also
- Concrete implementations in src/materials/
"""
abstract type AbstractMaterial end

"""
    AbstractElasticMaterial <: AbstractMaterial

Stateless elastic materials (no history variables).

Elastic materials compute stress directly from strain with no memory of loading history.

# Characteristics
- No internal state variables
- Reversible deformation
- Path-independent response
- `state_new = state_old` always

# Examples
- `LinearElastic`: Hooke's law (small strain)
- `NeoHookean`: Hyperelastic (finite strain)
- `Mooney-Rivlin`: Hyperelastic with two parameters
- `Ogden`: Hyperelastic for rubber-like materials

# See Also
- [`AbstractPlasticMaterial`](@ref) for history-dependent materials
"""
abstract type AbstractElasticMaterial <: AbstractMaterial end

"""
    AbstractPlasticMaterial <: AbstractMaterial

Stateful plastic materials (history-dependent).

Plastic materials have internal state variables that evolve with loading history.

# Characteristics
- Internal state variables (plastic strain, hardening, etc.)
- Irreversible deformation
- Path-dependent response
- `state_new ≠ state_old` during plastic loading

# Examples
- `PerfectPlasticity`: J2 plasticity with no hardening
- `IsotropicHardening`: J2 plasticity with isotropic hardening
- `KinematicHardening`: Bauschinger effect modeling
- `FiniteStrainPlasticity`: Large deformation plasticity

# State Variables

Common state variables:
- `εᵖ`: Plastic strain tensor
- `α`: Backstress (kinematic hardening)
- `κ`: Equivalent plastic strain (isotropic hardening)
- `damage`: Damage parameter (continuum damage mechanics)

# See Also
- [`AbstractElasticMaterial`](@ref) for stateless materials
"""
abstract type AbstractPlasticMaterial <: AbstractMaterial end

# ============================================================================
# MATERIAL MODEL FUNCTIONS
# ============================================================================

"""
    compute_stress(material::AbstractMaterial, ε, state_old, Δt) 
    -> (σ, 𝔻, state_new)

Compute stress, tangent, and updated state for a material model.

# Arguments
- `material`: Material model parameters
- `ε`: Strain tensor (SymmetricTensor{2,3} or similar)
- `state_old`: Previous state (Dict, NamedTuple, or nothing for elastic)
- `Δt`: Time step (for rate-dependent materials)

# Returns
- `σ`: Cauchy stress tensor
- `𝔻`: Material tangent (∂σ/∂ε)
- `state_new`: Updated internal state

# Examples

```julia
# Elastic material (no state)
σ, 𝔻, _ = compute_stress(LinearElastic(E=210e9, ν=0.3), ε, nothing, 0.0)

# Plastic material (with state)
state = (εᵖ=zero(SymmetricTensor{2,3}), κ=0.0)
σ, 𝔻, state_new = compute_stress(PerfectPlasticity(E=210e9, ν=0.3, σ_y=250e6), 
                                   ε, state, Δt)
```

# Implementation Notes

Material models implement this function with specific signatures:
- Elastic: `compute_stress(::LinearElastic, ε, _, _)`
- Plastic: `compute_stress(::PerfectPlasticity, ε, state, Δt)`

# See Also
- [`elasticity_tensor`](@ref) for elastic constitutive tensor
- Material implementations in src/materials/
"""
function compute_stress end

"""
    elasticity_tensor(material::AbstractElasticMaterial) -> Tensor{4,3}

Compute 4th-order elasticity tensor for an elastic material.

For linear elastic material:
```
C_ijkl = λ δ_ij δ_kl + μ (δ_ik δ_jl + δ_il δ_jk)
```

where:
- λ = Eν/((1+ν)(1-2ν)) (Lamé's first parameter)
- μ = E/(2(1+ν)) (shear modulus)

# Arguments
- `material`: Elastic material with parameters (E, ν, etc.)

# Returns
- `C`: 4th-order elasticity tensor (Tensor{4,3})

# Examples

```julia
mat = LinearElastic(E=210e9, ν=0.3)
C = elasticity_tensor(mat)

# Use in stress computation
σ = C ⊡ ε  # Double-dot product: σ_ij = C_ijkl ε_kl
```

# See Also
- [`compute_stress`](@ref) for full stress computation
"""
function elasticity_tensor end
