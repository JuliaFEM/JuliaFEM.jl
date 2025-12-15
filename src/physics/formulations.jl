# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Helper utilities for multi-field microkernel assembly.

Works with existing Element{K,P,S} system where S is a NamedTuple specifying fields.

# Philosophy

Element{K,P,S} encodes everything:
- K: Topology (Triangle, Tetrahedron, etc.)
- P: Basis (Lagrange{1}, Lagrange{2}, etc.)
- S: NamedTuple of fields with types and entity locations

# Example

```julia
# Thermoelasticity using Element{K,P,S} directly:
S = @DOFSet{T::DOF{Temperature,Vertex}, u::DOF{Displacement{3},Vertex}}
elem = Element{Tet4, Lagrange{1}, S}(id, dof_indices)

# Extract field info (compile-time!):
field_names = fieldnames(S)  # (:T, :u)
range_T = field_dof_range(elem, :T)  # Compile-time range
```
"""

using ..JuliaFEM: AbstractField, Displacement, Temperature

# ============================================================================
# FIELD TYPE EXTRACTION FROM S (Element{K,P,S})
# ============================================================================

"""
    field_type_for_dispatch(S, field_name::Symbol)

Extract field type from Element's S parameter for dispatch.

Works at runtime with NamedTuple-based field specifications.

# Example
```julia
S = @DOFSet{T::DOF{Temperature,Vertex}, u::DOF{Displacement{3},Vertex}}
field_type_for_dispatch(S, :T)  # Temperature()
field_type_for_dispatch(S, :u)  # Displacement{3}()
```
"""
function field_type_for_dispatch(::Type{S}, field_name::Symbol) where S<:DOFSet
    # Get field names at runtime
    names = fieldnames(S)
    idx = findfirst(==(field_name), names)
    isnothing(idx) && error("Field $field_name not found in $S")
    
    # Get the type of that field
    field_tuple_type = fieldtype(S, idx)
    
    # field_tuple_type should be DOF{FieldType, EntityType}
    # Extract the FieldType (first element)
    FieldType = field_tuple_type.parameters[1]
    
    # FieldType must be a field type (AbstractField)
    if FieldType <: AbstractField
        # Create an instance of the field type
        return FieldType()
    else
        error("Expected field type (AbstractField) in DOF, got $FieldType. Use format: DOF{FieldType, EntityType} where FieldType <: AbstractField")
    end
end

# ============================================================================
# CONVENIENCE TYPE ALIASES (for documentation/examples)
# ============================================================================

"""
    ThermoelasticityFields

Type alias for common thermoelasticity field specification.
Use as Element{K, P, ThermoelasticityFields} type parameter.

# Example
```julia
const ThermoelasticityFields = @DOFSet{T::DOF{Temperature,Vertex}, u::DOF{Displacement{3},Vertex}}
elem = Element{Tet4, Lagrange{1}, ThermoelasticityFields}(id, dofs)
```
"""
const ThermoelasticityFields = @DOFSet{T::DOF{Temperature,Vertex}, u::DOF{Displacement{3},Vertex}}
