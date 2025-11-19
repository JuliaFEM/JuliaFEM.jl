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
# MATERIAL BEHAVIOR TRAITS
# ============================================================================

"""
    MaterialBehavior

Abstract type for material behavior traits.

Material behavior traits allow integration code to query what computational
requirements a material has (constant vs strain-dependent tangent, stateless
vs stateful) without writing material-specific integration functions.

# Type Hierarchy
- `StatelessConstantTangent` - Tangent is constant (e.g., LinearElastic)
- `StatelessStrainDependent` - Tangent depends on strain (e.g., NeoHookean)
- `StatefulStrainDependent` - Has internal state variables (e.g., PerfectPlasticity)

# Usage
Each material declares its behavior via:
```julia
material_behavior(::MyMaterial) = StatelessConstantTangent()
```

Integration code can then dispatch on behavior:
```julia
behavior = material_behavior(material)
𝔻 = compute_tangent_at_point(material, behavior, ...)
```

# See Also
- [`material_behavior`](@ref) - Trait function
- [`needs_deformation`](@ref) - Query if material needs displacement field
- [`needs_state`](@ref) - Query if material has internal state
"""
abstract type MaterialBehavior end

"""
    StatelessConstantTangent <: MaterialBehavior

Material with constant tangent modulus (independent of strain).

Materials with this behavior:
- Compute tangent once (e.g., at reference strain E=0)
- Reuse same tangent at all integration points
- No displacement field needed during integration
- No internal state variables

**Examples:** LinearElastic

**Performance:** Fastest - tangent computed once per element
"""
struct StatelessConstantTangent <: MaterialBehavior end

"""
    StatelessStrainDependent <: MaterialBehavior

Material with strain-dependent tangent modulus (no internal state).

Materials with this behavior:
- Tangent depends on current strain/deformation
- Must compute tangent at each integration point
- Requires displacement field to compute deformation gradient F
- No internal state variables (path-independent)

**Examples:** NeoHookean, Mooney-Rivlin, Ogden

**Performance:** Moderate - tangent computed at each integration point
"""
struct StatelessStrainDependent <: MaterialBehavior end

"""
    StatefulStrainDependent <: MaterialBehavior

Material with strain-dependent tangent and internal state variables.

Materials with this behavior:
- Tangent depends on current strain and state history
- Must compute tangent at each integration point
- Requires displacement field to compute strain
- Has internal state variables (e.g., plastic strain, damage)
- History-dependent (path-dependent)

**Examples:** PerfectPlasticity, FiniteStrainPlasticity, ContinuumDamage

**Performance:** Slowest - tangent + state update at each integration point
"""
struct StatefulStrainDependent <: MaterialBehavior end

"""
    material_behavior(material::AbstractMaterial) -> MaterialBehavior

Trait function declaring what computational requirements a material has.

# Returns
- `StatelessConstantTangent()` - Constant tangent (e.g., LinearElastic)
- `StatelessStrainDependent()` - Strain-dependent tangent, no state (e.g., NeoHookean)
- `StatefulStrainDependent()` - Strain-dependent tangent + state (e.g., PerfectPlasticity)

# Examples
```julia
material_behavior(::LinearElastic) = StatelessConstantTangent()
material_behavior(::NeoHookean) = StatelessStrainDependent()
material_behavior(::PerfectPlasticity) = StatefulStrainDependent()
```

# Implementation Required
All concrete material types must implement this trait function.

# See Also
- [`MaterialBehavior`](@ref) - Behavior trait types
- [`needs_deformation`](@ref) - Convenience query
- [`needs_state`](@ref) - Convenience query
"""
function material_behavior end

"""
    needs_deformation(material::AbstractMaterial) -> Bool

Query whether material needs displacement field for tangent computation.

Returns `true` for materials with strain-dependent tangent, `false` for
materials with constant tangent.

# Examples
```julia
needs_deformation(LinearElastic(E=210e9, ν=0.3))      # false
needs_deformation(NeoHookean(μ=1e6, λ=1e9))           # true
needs_deformation(PerfectPlasticity(...))              # true
```
"""
needs_deformation(mat::AbstractMaterial) = !(material_behavior(mat) isa StatelessConstantTangent)

"""
    needs_state(material::AbstractMaterial) -> Bool

Query whether material has internal state variables.

Returns `true` for stateful materials (plasticity, damage), `false` for
stateless materials (elasticity).

# Examples
```julia
needs_state(LinearElastic(E=210e9, ν=0.3))      # false
needs_state(NeoHookean(μ=1e6, λ=1e9))           # false
needs_state(PerfectPlasticity(...))              # true
```
"""
needs_state(mat::AbstractMaterial) = material_behavior(mat) isa StatefulStrainDependent

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
