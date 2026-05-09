# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

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

All material models must implement `compute_stress(material, ε, state_old, Δt)`.
"""
abstract type AbstractMaterial end

"""
    AbstractElasticMaterial <: AbstractMaterial

Stateless elastic materials (no history variables).
"""
abstract type AbstractElasticMaterial <: AbstractMaterial end

"""
    AbstractPlasticMaterial <: AbstractMaterial

Stateful plastic materials (history-dependent).
"""
abstract type AbstractPlasticMaterial <: AbstractMaterial end

# Per-integration-point material state is represented as a `NamedTuple` whose
# field types are derived from `required_state_variables(material)`. The
# concrete `NamedTuple` type for a given material is produced by
# `material_state_type` (see `src/materials/global_material_cache.jl`).

# ============================================================================
# MATERIAL BEHAVIOR TRAITS
# ============================================================================

"""
    MaterialBehavior

Abstract type for material behavior traits.
"""
abstract type MaterialBehavior end

"""
    StatelessConstantTangent <: MaterialBehavior

Material whose tangent is **constant** with respect to strain at fixed temperature (etc.);
history plays no role and each integration point can reuse one cached `(σ, 𝔻)` pair on the
assembly hot path (see [`update_material_cache!`](@ref)).

Representative models: [`LinearElastic`](@ref), [`OrthotropicLinearElastic`](@ref),
[`HeatConductivity`](@ref), [`HydraulicConductivity`](@ref), [`MoistureDiffusivity`](@ref),
[`ElementWiseScalarDiffusion`](@ref).

Nonlinear diffusion coefficients still dispatch here only when implemented as a **piecewise
constant per element** cache (`ElementWiseScalarDiffusion`); strongly nonlinear κ(T) laws would
need either explicit differentiation or a strain-/temperature-dependent trait branch that does
not exist yet.
"""
struct StatelessConstantTangent <: MaterialBehavior end

"""
    StatelessStrainDependent <: MaterialBehavior

**Nonlinear elasticity or hyperelasticity**: tangent `𝔻` varies with strain (or deformation),
but there are **no persistent IP history variables** carried through [`GlobalMaterialCache`](@ref).

In continuum assembly, [`update_material_cache!`](@ref) always builds Green–Lagrange strain from
`F` at each integration point (even if the stress–strain map is written as a small-strain law).

Representative models: [`NeoHookean`](@ref), [`MooneyRivlin`](@ref), [`Yeoh3`](@ref), [`Gent`](@ref).

Ogden, Arruda–Boyce, etc., would also belong here once implemented.
"""
struct StatelessStrainDependent <: MaterialBehavior end

"""
    StatefulStrainDependent <: MaterialBehavior

**Incremental / path-dependent models**: stress and tangent depend on strain **and** on stored
state (`PlasticStrain`, `DamageVariable`, creep strain, eigenstrain, …). [`needs_state`](@ref)
is true; [`update_material_cache!`](@ref) reads [`get_old_state`](@ref) and writes back via
[`set_state!`](@ref).

Tangent semantics vary by model:

- [`PerfectPlasticity`](@ref), [`J2LinearIsotropicPlasticity`](@ref), [`StVenantKirchhoffJ2Plasticity`](@ref),
  [`ChabocheJ2Plasticity`](@ref): algorithmic elastoplastic tangents of the same *rank-one shear*
  class used in many textbooks (they are **not** the full spatial tangent unless augmented with
  the usual deviatoric projection terms).
- [`ScalarDamageLinearElastic`](@ref): explicit damage stagger; `𝔻` drops derivatives `∂d/∂ε`.
- [`NortonCreepElastic`](@ref): creep increment is explicit in `Δt`; returned `𝔻` is the elastic
  modulus only.
- [`LinearElasticWithEigenstrain`](@ref): `𝔻` is elastic; eigenstrain is updated outside
  `compute_stress`.

Representative solids: [`PerfectPlasticity`](@ref) (kinematic linear hardening),
[`J2LinearIsotropicPlasticity`](@ref) (linear isotropic expansion of the yield surface),
[`StVenantKirchhoffJ2Plasticity`](@ref), [`ScalarDamageLinearElastic`](@ref),
[`ChabocheJ2Plasticity`](@ref), [`NortonCreepElastic`](@ref),
[`LinearElasticWithEigenstrain`](@ref).

Still missing for broad classical coverage: pressure-dependent yield (Drucker–Prager,
Mohr–Coulomb), porous metal (Gurson), nonlinear hardening laws (Voce / Swift) as separate types,
rate-dependent Perzyna/overstress models, and extended hyperelastic families (Ogden).
"""
struct StatefulStrainDependent <: MaterialBehavior end

"""
    material_behavior(material::AbstractMaterial) -> MaterialBehavior

Trait function declaring what computational requirements a material has.
"""
function material_behavior end

"""
    needs_deformation(material::AbstractMaterial) -> Bool

Query whether material needs displacement field for tangent computation.
"""
needs_deformation(mat::AbstractMaterial) = !(material_behavior(mat) isa StatelessConstantTangent)

"""
    needs_state(material::AbstractMaterial) -> Bool

Query whether material has internal state variables.
"""
needs_state(mat::AbstractMaterial) = material_behavior(mat) isa StatefulStrainDependent

# ============================================================================
# MATERIAL MODEL FUNCTIONS
# ============================================================================

"""
    compute_stress(material::AbstractMaterial, ε, state_old, Δt)
    -> (σ, 𝔻, state_new)

Compute stress, tangent, and updated state for a material model.
"""
function compute_stress end

"""
    elasticity_tensor(material::AbstractElasticMaterial) -> Tensor{4,3}

Compute 4th-order elasticity tensor for an elastic material.
"""
function elasticity_tensor end
