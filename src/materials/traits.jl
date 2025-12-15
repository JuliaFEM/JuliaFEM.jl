# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Material trait functions using compositional state variable design.
"""

# Note: This file defines trait functions but doesn't implement them for specific materials.
# Material implementations (linear_elastic.jl, perfect_plasticity.jl, etc.) will
# implement these traits for their specific material types.

# ============================================================================
# PHYSICS SUPPORT TRAITS
# ============================================================================

"""
    supported_physics(material)

Returns tuple of physics types that this material supports.
"""
function supported_physics end

"""
    required_field_types(material)

Returns tuple of field types required by this material.
"""
function required_field_types(material)
    physics_tuple = supported_physics(material)
    # Extract field types from each physics
    map(p -> required_field_type(p), physics_tuple)
end

# ============================================================================
# STATE VARIABLE TRAITS
# ============================================================================

"""
    required_state_variables(material)

Returns tuple of state variable types required by this material.
"""
function required_state_variables end

# Type-level dispatch for required_state_variables (for @generated functions)
required_state_variables(::Type{T}) where {T<:AbstractMaterial} = required_state_variables(T())

# ============================================================================
# MATERIAL-SPECIFIC TRAIT IMPLEMENTATIONS
# ============================================================================

# NOTE: Trait implementations for specific materials (LinearElastic, PerfectPlasticity, etc.)
# are defined in their respective material files (linear_elastic.jl, perfect_plasticity.jl, etc.)
# This keeps the trait system extensible - new materials just implement these three traits:
#   - supported_physics(::MyMaterial)
#   - required_state_variables(::MyMaterial)
# and the rest (required_field_types, etc.) is automatically derived.

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

"""
    get_state_variable_types(material)

Returns tuple of concrete types for all state variables.
"""
function get_state_variable_types(material)
    vars = required_state_variables(material)
    map(v -> state_variable_type(v), vars)
end

"""
    get_state_variable_symbols(material)

Returns tuple of default symbols for all state variables.
"""
function get_state_variable_symbols(material)
    vars = required_state_variables(material)
    map(v -> default_symbol(v), vars)
end

"""
    is_stateful(material)

Returns true if material has state variables, false otherwise.
"""
is_stateful(material) = !isempty(required_state_variables(material))
