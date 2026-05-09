# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Assembly material workspace for zero-allocation assembly.

Defines `AssemblyMaterialWorkspace`, the per-element temporary workspace
used during assembly. Stores material fields (e.g. stress σ, tangent 𝔻)
and per-IP state for one element at a time and is reset between
elements.

This is different from `GlobalMaterialCache`, which stores persistent
state across all elements and time steps.
"""

using Tensors
using ..JuliaFEM: material_field_type, material_state_type, create_zero_field, create_zero_state

"""
    AssemblyMaterialWorkspace{FieldType, StateType}

Per-element temporary workspace for material fields and state during
assembly. Field structure (`FieldType`) is inferred from the material's
`supported_physics()` trait; state structure (`StateType`) from
`required_state_variables()`. Uses Array-of-Structs layout
(`Vector{NamedTuple}`) for cache-friendly per-IP access.

# Type Parameters
- `FieldType`: NamedTuple type for material fields, e.g. `(σ=..., 𝔻=...)` for mechanics.
- `StateType`: NamedTuple type for state, e.g. `(ε_p=..., α=..., κ=...)` for plasticity.

# Fields
- `fields::Vector{FieldType}`: material fields at each IP for ONE element.
- `states::Vector{StateType}`: temporary state at each IP for ONE element.

# Zero-Allocation Access

Direct per-IP field access:
```julia
workspace.fields[q].σ   # stress at IP q
workspace.fields[q].𝔻   # tangent at IP q
```

Per-IP write (reuse a pre-built NamedTuple in the hot loop):
```julia
fields_ref = (σ=σ_val, 𝔻=𝔻_val)
for q in 1:nips
    workspace.fields[q] = fields_ref
end
```

# See Also
- `GlobalMaterialCache`: persistent state storage for time-stepping.
- `get_tangent` / `get_stress` / `get_tangent_vector` / `get_stress_vector`:
  typed accessors used by domain kernels and tests.
- `extract_tangent!`: type-stable zero-allocation tangent extraction
  used by both COO and DOF-based assemblers.
"""
struct AssemblyMaterialWorkspace{FieldType<:NamedTuple, StateType<:NamedTuple} <: AbstractAssemblyMaterialWorkspace{FieldType, StateType}
    fields::Vector{FieldType}
    states::Vector{StateType}
end

# ============================================================================
# ZERO-ALLOCATION FIELD EXTRACTION HELPERS
# ============================================================================

"""
    get_tangent_vector(workspace::AssemblyMaterialWorkspace, buffer::Vector) -> Vector

Extract tangent vector from AoS structure using pre-allocated buffer (zero-allocation).

Updates buffer in-place and returns reference to buffer.
This eliminates allocations from Vector() constructor in list comprehension.

# Arguments
- `workspace`: Assembly material workspace
- `buffer`: Pre-allocated buffer (must have length >= length(workspace.fields))

# Returns
- Reference to buffer (updated in-place)

# Zero-Allocation
Direct assignment to buffer elements is zero-allocation (no Vector() constructor).
"""
@inline function get_tangent_vector(
    workspace::AssemblyMaterialWorkspace{FieldType},
    buffer::Vector{T}
) where {FieldType, T}
    if !hasfield(FieldType, :𝔻)
        error("FieldType $FieldType does not have :𝔻 field")
    end
    # Update buffer in-place (zero allocation - direct assignment)
    n = length(workspace.fields)
    @inbounds for i in 1:n
        buffer[i] = workspace.fields[i].𝔻
    end
    return buffer
end

"""
    get_tangent_vector(workspace::AssemblyMaterialWorkspace) -> Vector

Extract tangent vector from AoS structure (allocates new Vector).

DEPRECATED: Use `get_tangent_vector(workspace, buffer)` with pre-allocated buffer
for zero-allocation access.

This version allocates a new Vector via list comprehension.
"""
function get_tangent_vector(workspace::AssemblyMaterialWorkspace{FieldType}) where {FieldType}
    if !hasfield(FieldType, :𝔻)
        error("FieldType $FieldType does not have :𝔻 field")
    end
    # Extract vector by accessing each field's 𝔻 component
    # This allocates once when called, but is outside the hot loop
    return [workspace.fields[i].𝔻 for i in 1:length(workspace.fields)]
end

"""
    get_stress_vector(workspace::AssemblyMaterialWorkspace, buffer::Vector) -> Vector

Extract stress vector from AoS structure using pre-allocated buffer
(zero-allocation).

Updates buffer in-place and returns the same buffer.
"""
@inline function get_stress_vector(
    workspace::AssemblyMaterialWorkspace{FieldType},
    buffer::Vector{T}
) where {FieldType, T}
    if !hasfield(FieldType, :σ)
        error("FieldType $FieldType does not have :σ field")
    end
    n = length(workspace.fields)
    @inbounds for i in 1:n
        buffer[i] = workspace.fields[i].σ
    end
    return buffer
end

"""
    get_stress_vector(workspace::AssemblyMaterialWorkspace) -> Vector

Extract stress vector from AoS structure.
Creates vector by extracting σ from each field - called once outside hot loop.
"""
function get_stress_vector(workspace::AssemblyMaterialWorkspace{FieldType}) where {FieldType}
    if !hasfield(FieldType, :σ)
        error("FieldType $FieldType does not have :σ field")
    end
    # Extract vector by accessing each field's σ component
    # This allocates once when called, but is outside the hot loop
    return [workspace.fields[i].σ for i in 1:length(workspace.fields)]
end

# ============================================================================
# ZERO-ALLOCATION FIELD ACCESSORS
# ============================================================================
#
# `workspace.fields` and `workspace.states` resolve via Julia's default
# `getproperty` since they are real struct fields. Per-IP, per-field
# access goes through the typed accessors below (`get_stress`,
# `get_tangent`, `get_field`) which use compile-time field-index lookup
# in `@generated` bodies for zero-allocation reads.

"""
    get_stress(workspace::AssemblyMaterialWorkspace, ip::Int)

Get stress tensor at integration point `ip` (mechanics-only fields).
Uses compile-time field-index lookup, zero-allocation.

# Examples
```julia
workspace = create_material_cache(LinearElastic(...), 8)
σ = get_stress(workspace, 1)
```
"""
@generated function get_stress(workspace::AssemblyMaterialWorkspace{FieldType}, ip::Int) where {FieldType}
    # Check if FieldType has :σ field
    if hasfield(FieldType, :σ)
        # Get field type for type stability
        field_type = fieldtype(FieldType, :σ)
        
        # Find field index in FieldType NamedTuple
        field_names = fieldnames(FieldType)
        σ_idx = findfirst(==(:σ), field_names)
        if σ_idx === nothing
            error("FieldType $FieldType does not have :σ field")
        end
        
        # Access via fields[ip].σ - zero allocation (compile-time known indices)
        return :(@inbounds return getfield(workspace.fields[ip], $σ_idx)::$field_type)
    else
        error("FieldType $FieldType does not have :σ field")
    end
end

"""
    get_tangent(workspace::AssemblyMaterialWorkspace, ip::Int)

Get tangent modulus from workspace (mechanics only).

# Backward Compatibility
Replaces `workspace.𝔻[ip]` with `get_tangent(workspace, ip)`.

# Examples
```julia
workspace = create_material_cache(LinearElastic(...), 8)
𝔻 = get_tangent(workspace, 1)  # → Tangent at IP 1
```
"""
# Accessor using @generated for compile-time field lookup
# ============================================================================
# TYPE-STABLE FIELD ACCESS HELPERS (Zero-Allocation)
# ============================================================================

"""
    @generated function _get_tangent_field_index(::Type{FieldType}) where {FieldType<:NamedTuple}

Get compile-time field index for `:𝔻` field in FieldType.

Returns the field index as a compile-time constant, enabling type-stable `getfield` access.
"""
@generated function _get_tangent_field_index(::Type{FieldType}) where {FieldType<:NamedTuple}
    field_names = fieldnames(FieldType)
    field_idx = findfirst(==(:𝔻), field_names)
    
    if field_idx === nothing
        error("FieldType $FieldType does not have field :𝔻")
    end
    
    # Return the compile-time constant index
    return field_idx
end

"""
    extract_tangent!(buffer::AbstractVector{SymmetricTensor{4,3,Float64,36}},
                     fields::Vector{FieldType},
                     ::Type{FieldType}) where {FieldType<:NamedTuple}

Extract tangent field `:𝔻` from fields vector into buffer (type-stable, zero-allocation).

Uses compile-time field index lookup to avoid Symbol-based getfield which causes type instability.

`buffer` is `AbstractVector` so the DOF-based assembler can pass either
a plain `Vector{Buf}` or a column view into a `Matrix{Buf}` without
copying.
"""
@inline function extract_tangent!(
    buffer::AbstractVector{SymmetricTensor{4,3,Float64,36}},
    fields::Vector{FieldType},
    ::Type{FieldType}
) where {FieldType<:NamedTuple}
    # Get compile-time field index for :𝔻
    field_idx = _get_tangent_field_index(FieldType)  # Compile-time constant!
    n = length(fields)
    @inbounds for i in 1:n
        # Use compile-time known index - type-stable and zero-allocation
        buffer[i] = getfield(fields[i], field_idx)::SymmetricTensor{4,3,Float64,36}
    end
    return nothing
end

@generated function get_tangent(workspace::AssemblyMaterialWorkspace{FieldType}, ip::Int) where {FieldType}
    # Check if FieldType has :𝔻 field
    if hasfield(FieldType, :𝔻)
        # Get field type for type stability
        field_type = fieldtype(FieldType, :𝔻)
        
        # Find field index in FieldType NamedTuple
        field_names = fieldnames(FieldType)
        𝔻_idx = findfirst(==(:𝔻), field_names)
        if 𝔻_idx === nothing
            error("FieldType $FieldType does not have :𝔻 field")
        end
        
        # Access via fields[ip].𝔻 - zero allocation (compile-time known indices)
        return :(@inbounds return getfield(workspace.fields[ip], $𝔻_idx)::$field_type)
    else
        error("FieldType $FieldType does not have :𝔻 field")
    end
end

# set_fields! - CRITICAL: This function MUST be zero-allocation
# Uses @generated function to generate code that constructs NamedTuple at compile time
# The generated code uses getfield with compile-time indices to extract values
# Compiler should optimize NamedTuple construction to zero allocation
@generated function set_fields!(workspace::AssemblyMaterialWorkspace{FieldType}, ip::Int, field_values::NamedTuple) where {FieldType}
    field_names = fieldnames(FieldType)
    n_fields = length(field_names)
    
    # Generate code that extracts values using getfield with compile-time indices
    # This avoids runtime property access overhead
    field_accesses = [:(getfield(field_values, $i)) for i in 1:n_fields]
    
    # Construct NamedTuple using compile-time known structure
    # The compiler should optimize this to zero allocation if:
    # 1. FieldType is known at compile time (it is, via @generated)
    # 2. Field values are already allocated (they are, from compute_stress)
    # 3. NamedTuple wrapper can be optimized away (compiler optimization)
    names_tuple = Expr(:tuple, [QuoteNode(n) for n in field_names]...)
    values_tuple = Expr(:tuple, field_accesses...)
    
    # Generate: workspace.fields[ip] = NamedTuple{(:σ, :𝔻)}((σ_val, 𝔻_val))
    # This should be zero-allocation after compiler optimization
    return :(@inbounds workspace.fields[ip] = NamedTuple{$names_tuple}($values_tuple); return nothing)
end

"""
    reset!(workspace::AssemblyMaterialWorkspace)

Reset assembly material workspace to zero values.

# Side Effects
Mutates all arrays in workspace to zero.
"""
function reset!(workspace::AssemblyMaterialWorkspace{FieldType, StateType}) where {FieldType, StateType}
    # `getfield` is type-stable; `getproperty` would allocate inside the loop.
    fields = getfield(workspace, 1)
    states = getfield(workspace, 2)
    # Building the zero values per call is fine here — `reset!` is called once
    # per element, not per integration point. The hot-loop variant below skips
    # this by letting the caller pass them in.
    zero_field = create_zero_field(FieldType)
    zero_state = create_zero_state(StateType)
    n = length(fields)
    @inbounds for i in 1:n
        fields[i] = zero_field
        states[i] = zero_state
    end
    return nothing
end

# Zero-allocation overload: caller hands in the (already pre-allocated) zero
# values, so the loop below is allocation-free.
function reset!(
    workspace::AssemblyMaterialWorkspace{FieldType, StateType},
    zero_field::FieldType,
    zero_state::StateType,
) where {FieldType, StateType}
    fields = getfield(workspace, 1)
    states = getfield(workspace, 2)
    n = length(fields)
    @inbounds for i in 1:n
        fields[i] = zero_field
        states[i] = zero_state
    end
    return nothing
end

# ============================================================================
# CONSTRUCTORS
# ============================================================================

"""
    create_material_cache(material::M, max_nips::Int) -> AssemblyMaterialWorkspace{FieldType, StateType}
        where {M <: AbstractMaterial}

Create pre-allocated assembly material workspace with field and state types inferred from material traits.

Uses trait system to determine:
- `FieldType` from `material_field_type(material)` (inferred from `supported_physics()`)
- `StateType` from `material_state_type(material)` (inferred from `required_state_variables()`)

Purpose: Create temporary workspace for ONE element during assembly.
Note: For persistent state storage, use `create_global_material_cache()` instead.

# Arguments
- `material`: Material model (type M determines field and state types)
- `max_nips`: Maximum integration points per element

# Returns
- `AssemblyMaterialWorkspace{FieldType, StateType}` with field structure inferred from material

# Type Stability
Return type is fully inferrable:
- `M` is concrete material type (known at compile time)
- `FieldType = material_field_type(material)` is concrete NamedTuple type (trait dispatch)
- `StateType = material_state_type(material)` is concrete NamedTuple type (trait dispatch)
- Zero allocations in hot loops!

# Examples
```julia
# Stateless material (mechanics)
mat = LinearElastic(E=210e9, ν=0.3)
workspace = create_material_cache(mat, 8)
# → AssemblyMaterialWorkspace{(:σ, :𝔻), ()}
workspace.fields[1].σ  # → Stress
workspace.fields[1].𝔻  # → Tangent

# Stateful material (mechanics with plasticity)
mat = PerfectPlasticity(E=210e9, ν=0.3, σ_y=250e6)
workspace = create_material_cache(mat, 8)
# → AssemblyMaterialWorkspace{(:σ, :𝔻), (:ε_p, :α, :κ)}
workspace.fields[1].σ  # → Stress
workspace.states[1]    # → (ε_p=..., α=..., κ=...)
```

# See Also
- `create_global_material_cache()`: For persistent state storage (all elements, time-stepping)
- `material_field_type()`: Trait function to infer field structure
"""
function create_material_cache(material::M, max_nips::Int) where M<:AbstractMaterial
    # Infer field type from material traits
    FieldType = material_field_type(material)
    StateType = material_state_type(material)
    
    # Create zero-initialized field NamedTuple
    zero_field = create_zero_field(FieldType)
    
    # Create Vector of NamedTuples - one per integration point (AoS pattern)
    # This matches the prototype's MaterialContext pattern
    fields = [zero_field for _ in 1:max_nips]
    
    # Create zero-initialized states
    zero_state = create_zero_state(StateType)
    states = [zero_state for _ in 1:max_nips]
    
    return AssemblyMaterialWorkspace{FieldType, StateType}(fields, states)
end
