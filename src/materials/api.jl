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

"""
    AbstractMaterialState

Abstract type for material internal state at integration points.

Concrete types must be immutable for thread safety.
"""
abstract type AbstractMaterialState end

"""
    EmptyState <: AbstractMaterialState

Empty material state for stateless materials.
"""
struct EmptyState <: AbstractMaterialState end

# Zero constructor for Base.zero compatibility
Base.zero(::Type{EmptyState}) = EmptyState()

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

Material with constant tangent modulus (independent of strain).
"""
struct StatelessConstantTangent <: MaterialBehavior end

"""
    StatelessStrainDependent <: MaterialBehavior

Material with strain-dependent tangent modulus (no internal state).
"""
struct StatelessStrainDependent <: MaterialBehavior end

"""
    StatefulStrainDependent <: MaterialBehavior

Material with strain-dependent tangent and internal state variables.
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

"""
    state_type(::Type{<:AbstractMaterial}) -> Type{<:AbstractMaterialState}

Return the concrete state type for a given material type.
"""
state_type(::Type{<:AbstractMaterial}) = EmptyState

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
