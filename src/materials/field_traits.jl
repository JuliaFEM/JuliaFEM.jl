# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Material field trait functions for compositional field design.
"""

using Tensors
using ..JuliaFEM: AbstractPhysics, Elasticity, Thermal

# ============================================================================
# MATERIAL FIELD TRAITS
# ============================================================================

"""
    required_material_fields(physics::AbstractPhysics) -> Type{<:NamedTuple}

Returns NamedTuple type defining material fields required by this physics.
"""
function required_material_fields end

# ============================================================================
# PHYSICS-SPECIFIC IMPLEMENTATIONS
# ============================================================================

"""
    required_material_fields(::Elasticity{Dim}) where Dim

Elasticity physics requires stress (σ) and tangent modulus (𝔻).
"""
function required_material_fields(::Elasticity{Dim}) where Dim
    if Dim == 3
        return NamedTuple{
            (:σ, :𝔻),
            Tuple{
                SymmetricTensor{2,3,Float64,6},
                SymmetricTensor{4,3,Float64,36}
            }
        }
    elseif Dim == 2
        return NamedTuple{
            (:σ, :𝔻),
            Tuple{
                SymmetricTensor{2,2,Float64,3},
                SymmetricTensor{4,2,Float64,9}
            }
        }
    else
        error("Elasticity{Dim} with Dim=$Dim not supported")
    end
end

"""
    required_material_fields(::Thermal{Dim}) where Dim

Thermal physics requires heat flux (q) and thermal conductivity (k).
"""
function required_material_fields(::Thermal{Dim}) where Dim
    if Dim == 3
        return NamedTuple{
            (:q, :k),
            Tuple{
                Vec{3,Float64},
                SymmetricTensor{2,3,Float64,6}
            }
        }
    elseif Dim == 2
        return NamedTuple{
            (:q, :k),
            Tuple{
                Vec{2,Float64},
                SymmetricTensor{2,2,Float64,3}
            }
        }
    else
        error("Thermal{Dim} with Dim=$Dim not supported")
    end
end

# ============================================================================
# FIELD TYPE INFERENCE
# ============================================================================

"""
    material_field_type(material::AbstractMaterial) -> Type{<:NamedTuple}

Infer material field NamedTuple type from material's supported physics.
"""
function material_field_type(material::AbstractMaterial)
    physics_tuple = supported_physics(material)
    
    if isempty(physics_tuple)
        # No physics - empty fields
        return NamedTuple{(),Tuple{}}
    elseif length(physics_tuple) == 1
        # Single physics - direct mapping
        return required_material_fields(physics_tuple[1])
    else
        # Multiple physics - compose fields
        field_types = map(p -> required_material_fields(p), physics_tuple)
        return compose_field_types(field_types...)
    end
end

"""
    compose_field_types(nt_types::Type{<:NamedTuple}...) -> Type{<:NamedTuple}

Compose multiple NamedTuple types into one.
"""
@generated function compose_field_types(nt_types::Type{<:NamedTuple}...)
    # Collect all field names and types
    all_names = Symbol[]
    all_types = Type[]
    
    for nt_type in nt_types.parameters
        names = fieldnames(nt_type)
        types = [fieldtype(nt_type, name) for name in names]
        append!(all_names, names)
        append!(all_types, types)
    end
    
    # Build composed NamedTuple type
    names_tuple = Tuple(all_names)
    types_tuple = Tuple(all_types)
    
    return :(NamedTuple{$names_tuple, $types_tuple})
end

"""
    create_zero_field(::Type{FieldType}) where {FieldType<:NamedTuple}

Create zero-initialized field NamedTuple.
"""
@generated function create_zero_field(::Type{FieldType}) where {FieldType<:NamedTuple}
    names = fieldnames(FieldType)
    types = [fieldtype(FieldType, name) for name in names]
    zero_values = [:(zero($T)) for T in types]
    
    return :(NamedTuple{$names}(($(zero_values...),)))
end
