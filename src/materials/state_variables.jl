# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
State variable types for material models.
"""

using Tensors

# ============================================================================
# ABSTRACT STATE VARIABLE TYPE
# ============================================================================

"""
    AbstractStateVariable

Abstract type for material state variables.
"""
abstract type AbstractStateVariable end

# ============================================================================
# COMMON STATE VARIABLES
# ============================================================================

"""
    PlasticStrain <: AbstractStateVariable

Plastic strain tensor for plasticity models.

Concrete type: `SymmetricTensor{2,3,Float64,6}`
Default symbol: `:ε_p`
"""
struct PlasticStrain <: AbstractStateVariable end

"""
    Backstress <: AbstractStateVariable

Backstress tensor for kinematic hardening.

Concrete type: `SymmetricTensor{2,3,Float64,6}`
Default symbol: `:α`
"""
struct Backstress <: AbstractStateVariable end

"""
    EquivalentPlasticStrain <: AbstractStateVariable

Equivalent (accumulated) plastic strain for isotropic hardening.

Concrete type: `Float64`
Default symbol: `:κ`
"""
struct EquivalentPlasticStrain <: AbstractStateVariable end

"""
    DamageVariable <: AbstractStateVariable

Damage variable for continuum damage mechanics.

Concrete type: `Float64` (typically in range [0,1])
Default symbol: `:d`
"""
struct DamageVariable <: AbstractStateVariable end

# ============================================================================
# STATE VARIABLE TRAITS
# ============================================================================

"""
    state_variable_type(::Type{<:AbstractStateVariable})

Returns the concrete data type for a state variable.
"""
function state_variable_type end

state_variable_type(::Type{PlasticStrain}) = SymmetricTensor{2,3,Float64,6}
state_variable_type(::Type{Backstress}) = SymmetricTensor{2,3,Float64,6}
state_variable_type(::Type{EquivalentPlasticStrain}) = Float64
state_variable_type(::Type{DamageVariable}) = Float64

"""
    default_symbol(::Type{<:AbstractStateVariable})

Returns the default symbol name for a state variable.
"""
function default_symbol end

default_symbol(::Type{PlasticStrain}) = :ε_p
default_symbol(::Type{Backstress}) = :α
default_symbol(::Type{EquivalentPlasticStrain}) = :κ
default_symbol(::Type{DamageVariable}) = :d
