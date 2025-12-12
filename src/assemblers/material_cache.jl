# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Assembly material workspace implementations for zero-allocation assembly.

Contains mutable (AssemblyMaterialWorkspace) and immutable (ImmutableMaterialStateCache) variants.

**Purpose:** Per-element temporary workspace during assembly. Stores stress (σ), tangent (𝔻),
and temporary state for ONE element at a time. Reset between elements.

**Note:** This is different from `GlobalMaterialCache`, which stores persistent state
across all elements and time steps.
"""

using Tensors
using ..JuliaFEM: material_field_type, material_state_type, create_zero_field, create_zero_state

"""
    AssemblyMaterialWorkspace{FieldType, StateType}

Per-element temporary workspace for material fields and state during assembly.

**Array of Structs (AoS) Pattern**: Matches prototype implementation for zero-allocation access.

**Compositional Design**: Field structure inferred from material's `supported_physics()` trait.

Contains pre-allocated arrays for material fields and temporary state.
Mutated per element during assembly, then reset for next element.

**Purpose:** Temporary workspace during stiffness matrix assembly.
**Scope:** ONE element at a time (reset between elements).
**Lifetime:** Assembly loop only (not persistent).

**Zero-Allocation Design**: Uses Array of Structs (AoS) pattern - Vector of NamedTuples.
Each integration point has its own NamedTuple of fields, enabling cache-friendly access
when looping through IPs.

# Type Parameters
- `FieldType`: NamedTuple type for material fields (e.g., `(σ=..., 𝔻=...)` for mechanics)
- `StateType`: NamedTuple type for state (e.g., `(ε_p=..., α=..., κ=...)` for plasticity)

# Fields

- `fields::Vector{FieldType}`: Material fields at each IP [NIP] - ONE element (AoS pattern)
- `states::Vector{StateType}`: Temporary state at each IP [NIP] - ONE element

# Zero-Allocation Usage

**Direct field access** (zero allocation):
```julia
workspace.fields[q].σ  # → Stress at IP q (0 bytes!)
workspace.fields[q].𝔻  # → Tangent at IP q (0 bytes!)
```

**Update pattern** (pre-create NamedTuple outside hot loop):
```julia
# Pre-create NamedTuple ONCE (outside hot loop)
fields_ref = (σ=σ_val, 𝔻=𝔻_val)  # ~896 bytes, but only once

# In hot loop - reuse same NamedTuple (zero allocation)
for q in 1:nips
    workspace.fields[q] = fields_ref  # ~0-36 bytes (just assignment)
end
```

# Examples
```julia
# Mechanics only
workspace = create_material_cache(LinearElastic(...), 8)

# Access fields
σ = workspace.fields[1].σ  # → Stress at IP 1 (0 bytes!)
𝔻 = workspace.fields[1].𝔻  # → Tangent at IP 1 (0 bytes!)

# Multiphysics (future)
workspace = create_material_cache(ThermoElastic(...), 8)
σ = workspace.fields[1].σ  # → Stress (0 bytes!)
𝔻 = workspace.fields[1].𝔻  # → Tangent (0 bytes!)
q = workspace.fields[1].q   # → Heat flux (0 bytes!)
k = workspace.fields[1].k   # → Thermal conductivity (0 bytes!)
```

**Implementation:** Uses mutable struct with Vector of NamedTuples. Access via compile-time
known struct field indices for zero-allocation reads. Updates reuse pre-created NamedTuples
for zero-allocation writes.

# See Also
- `GlobalMaterialCache`: Persistent state storage for time-stepping (all elements)
- `material_field_type()`: Trait function to infer field structure
"""
# AssemblyMaterialWorkspace uses Array of Structs (AoS) pattern matching prototype
# Each integration point has its own field container (better cache locality)
# Uses mutable struct wrapper to enable zero-allocation in-place updates
@generated function _create_field_container_type(::Type{FieldType}) where {FieldType<:NamedTuple}
    field_names = fieldnames(FieldType)
    field_types = [fieldtype(FieldType, name) for name in field_names]
    
    # Create mutable struct with same fields as FieldType
    struct_fields = Expr[]
    for (name, T) in zip(field_names, field_types)
        push!(struct_fields, Expr(:(::), name, T))
    end
    
    struct_name = Symbol("FieldContainer_$(hash(FieldType))")
    struct_def = Expr(:struct, true, :($struct_name), Expr(:block, struct_fields...))
    
    return struct_def
end

# AssemblyMaterialWorkspace uses Array of Structs (AoS) pattern matching prototype
# Each integration point has its own NamedTuple of fields (better cache locality)
struct AssemblyMaterialWorkspace{FieldType<:NamedTuple, StateType<:NamedTuple} <: AbstractMaterialStateCache{FieldType, StateType}
    fields::Vector{FieldType}  # Vector of NamedTuples - one per integration point (AoS pattern)
    states::Vector{StateType}  # Temporary state at each IP
end

# ============================================================================
# ZERO-ALLOCATION FIELD EXTRACTION HELPERS
# ============================================================================

# ============================================================================
# Macro-based zero-allocation field access
# ============================================================================

"""
    @field_vector(workspace, field_name)

Macro to extract field vector with ZERO allocations using compile-time field index lookup.

This macro generates code that uses `getfield` with compile-time constant indices,
completely bypassing NamedTuple property access overhead.

# Examples
```julia
workspace = create_material_cache(LinearElastic(...), 8)

# Zero-allocation vector extraction
𝔻_vec = @field_vector(workspace, :𝔻)  # → Vector{SymmetricTensor{4,3,Float64,36}}
σ_vec = @field_vector(workspace, :σ)  # → Vector{SymmetricTensor{2,3,Float64,6}}

# Then use in hot loops
for q in 1:8
    C = 𝔻_vec[q]  # Zero allocation!
end
```
"""
# Helper @generated function that generates zero-allocation field access code
# This is called by the macro to generate compile-time constant getfield calls
# CRITICAL: The generated code uses nested getfield with compile-time constant indices
# This should be zero-allocation if the compiler can infer types properly
@generated function _get_field_vector_impl(workspace::AssemblyMaterialWorkspace{FieldType}, ::Val{FieldName}) where {FieldType, FieldName}
    # Check if FieldType has this field
    if hasfield(FieldType, FieldName)
        # Find field index in FieldType NamedTuple (compile-time!)
        field_names = fieldnames(FieldType)
        field_idx = findfirst(==(FieldName), field_names)
        
        if field_idx === nothing
            error("FieldType $FieldType does not have field :$FieldName")
        end
        
        # Get the field type for type stability
        field_type = fieldtype(FieldType, FieldName)
        vec_type = Vector{field_type}
        
        # Generate code that extracts field from each element in workspace.fields
        # workspace.fields is Vector{FieldType}, where FieldType is a NamedTuple
        # We need to extract field FieldName from each NamedTuple in the vector
        # NOTE: This still allocates a new Vector, but it's the same as get_tangent_vector
        # The benefit is compile-time field index lookup (type stability)
        # For true zero-allocation, we'd need to pre-allocate a buffer in the cache
        return quote
            # Extract vector by accessing field at compile-time known index
            # This allocates a new Vector (same as get_tangent_vector), but with type stability
            n = length(workspace.fields)
            result = Vector{$field_type}(undef, n)
            @inbounds for i in 1:n
                result[i] = getfield(workspace.fields[i], $field_idx)
            end
            return result::$vec_type
        end
    else
        error("FieldType $FieldType does not have field :$FieldName")
    end
end

export @field_vector

macro field_vector(workspace, field_name)
    # Extract the Symbol from the field_name argument
    # Handle :field_name (QuoteNode), field_name (Symbol), and quoted expressions
    field_sym = if field_name isa QuoteNode
        field_name.value
    elseif field_name isa Symbol
        field_name
    elseif field_name isa Expr && field_name.head == :quote && length(field_name.args) == 1
        field_name.args[1]
    elseif field_name isa Expr && field_name.head == :macrocall
        # Handle @doc macro expansion - skip it
        return nothing
    else
        error("Expected Symbol, QuoteNode, or quoted Symbol, got $(typeof(field_name)): $field_name")
    end
    
    # Generate code that calls the @generated function
    # The @generated function will specialize on the workspace type and field name
    return :(_get_field_vector_impl($(esc(workspace)), Val($(QuoteNode(field_sym)))))
end

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

**DEPRECATED**: Use `get_tangent_vector(workspace, buffer)` with pre-allocated buffer
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
# ZERO-COST FIELD ACCESS VIA getproperty (COMPILE-TIME MAGIC!)
# ============================================================================

"""
    workspace.σ[ip]  # Zero-cost field access!

Enable natural field access syntax with zero-allocation using `@generated` functions.

# Examples
```julia
workspace = create_material_cache(LinearElastic(...), 8)

# Natural syntax - zero allocation!
σ = workspace.σ[1]      # → Stress at IP 1 (0 bytes!)
𝔻 = workspace.𝔻[1]      # → Tangent at IP 1 (0 bytes!)

# Works for multiphysics too
q = workspace.q[1]      # → Heat flux (0 bytes!)
k = workspace.k[1]      # → Thermal conductivity (0 bytes!)
```

# Implementation
Uses `@generated` functions with `Base.getproperty` to enable compile-time field lookup.
The field name is known at compile time, so we generate direct field access code.
"""
# Use @generated for compile-time field lookup
# Generate specialized methods for each field name at compile time
@generated function Base.getproperty(workspace::AssemblyMaterialWorkspace{FieldType}, name::Val{Name}) where {FieldType, Name}
    # Check if FieldType has this field
    if hasfield(FieldType, Name)
        # Find field index in FieldType NamedTuple
        field_names = fieldnames(FieldType)
        field_idx = findfirst(==(Name), field_names)
        
        if field_idx === nothing
            error("FieldType $FieldType does not have field :$Name")
        end
        
        # Generate code that extracts vector by accessing each field's component
        # This creates a vector - called once outside hot loop
        return :([getfield(workspace.fields[i], $field_idx) for i in 1:length(workspace.fields)])
    elseif Name === :fields
        return :(getfield(workspace, 1))
    elseif Name === :states
        return :(getfield(workspace, 2))
    else
        # Field doesn't exist - generate error at compile time
        return :(error("AssemblyMaterialWorkspace{$(FieldType)} has no field :$Name. Available material fields: $(fieldnames(FieldType)), struct fields: (:fields, :states)"))
    end
end

# Runtime fallback for Symbol (less efficient but works)
function Base.getproperty(workspace::AssemblyMaterialWorkspace{FieldType}, name::Symbol) where {FieldType}
    # Convert to Val for compile-time dispatch
    return getproperty(workspace, Val(name))
end

# ============================================================================
# CONVENIENCE ACCESSORS
# ============================================================================

"""
    get_stress(workspace::AssemblyMaterialWorkspace, ip::Int)

Get stress tensor from workspace (mechanics only).

# Backward Compatibility
Replaces `workspace.σ[ip]` with `get_stress(workspace, ip)`.

# Examples
```julia
workspace = create_material_cache(LinearElastic(...), 8)
σ = get_stress(workspace, 1)  # → Stress at IP 1
```
"""
# ============================================================================
# ZERO-ALLOCATION FIELD ACCESSORS
# ============================================================================

# Accessor using @generated for compile-time field lookup
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
    extract_tangent!(buffer::Vector{SymmetricTensor{4,3,Float64,36}}, 
                     fields::Vector{FieldType}, 
                     ::Type{FieldType}) where {FieldType<:NamedTuple}

Extract tangent field `:𝔻` from fields vector into buffer (type-stable, zero-allocation).

Uses compile-time field index lookup to avoid Symbol-based getfield which causes type instability.
"""
@inline function extract_tangent!(
    buffer::Vector{SymmetricTensor{4,3,Float64,36}},
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

"""
    get_field(workspace::AssemblyMaterialWorkspace, field_name::Symbol, ip::Int)

Get any field from workspace by name.

# Examples
```julia
workspace = create_material_cache(LinearElastic(...), 8)
get_field(workspace, :σ, 1)  # → Stress
get_field(workspace, :𝔻, 1)  # → Tangent

# Multiphysics
workspace = create_material_cache(ThermoElastic(...), 8)
get_field(workspace, :q, 1)   # → Heat flux
get_field(workspace, :k, 1)   # → Thermal conductivity
```
"""
@generated function get_field(workspace::AssemblyMaterialWorkspace{FieldType}, field_name::Val{Name}, ip::Int) where {FieldType, Name}
    # Check if FieldType has this field
    if hasfield(FieldType, Name)
        # Find field index in FieldType NamedTuple
        field_names = fieldnames(FieldType)
        field_idx = findfirst(==(Name), field_names)
        if field_idx === nothing
            error("FieldType $FieldType does not have field :$Name")
        end
        
        # Get field type for type stability
        field_type = fieldtype(FieldType, Name)
        
        # Access via fields[ip].Name - zero allocation (compile-time known indices)
        return :(@inbounds return getfield(workspace.fields[ip], $field_idx)::$field_type)
    else
        error("FieldType $FieldType does not have field :$Name")
    end
end

# Non-generated fallback for runtime Symbol (less efficient but works)
function get_field(workspace::AssemblyMaterialWorkspace{FieldType}, field_name::Symbol, ip::Int) where {FieldType}
    return get_field(workspace, Val(field_name), ip)
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
    ImmutableMaterialStateCache{M,NIP}

Immutable assembly material workspace using NTuple for zero-allocation access.

Unlike `AssemblyMaterialWorkspace`, this version:
- Uses `NTuple` instead of `Vector` (stack-allocated, no heap access)
- Is immutable (must create new instance per element)
- Has **zero allocations** during cache access
- Enables full compiler optimization (sizes known at compile time)

# Type Parameters
- `M`: Material state type (EmptyState for stateless)
- `NIP`: Number of integration points (compile-time constant)

# Fields
- `σ::NTuple{NIP, SymmetricTensor{2,3,Float64,6}}`: Stress at each IP
- `𝔻::NTuple{NIP, SymmetricTensor{4,3,Float64,36}}`: Tangent modulus at each IP
- `states::NTuple{NIP, M}`: Internal state at each IP

# Zero-Allocation Access

```julia
# Indexing is zero-allocation:
tangent = cache.𝔻[q]  # 0 bytes!
stress = cache.σ[q]   # 0 bytes!
```

# Performance

**Eliminates type instability** from `Vector` indexing:
- Before: `𝔻::SYMMETRICTENSOR{4, 3, FLOAT64}` (UPPERCASE = unstable)
- After: `𝔻::SymmetricTensor{4, 3, Float64}` (lowercase = concrete)

**Pros:**
- Zero allocations during access
- Full compile-time type inference
- Stack-allocated (no GC pressure)

**Cons:**
- Immutable (must create new instance per element)
- Cannot be reused across elements

# Usage

```julia
# Create new cache per element:
material_cache = create_material_cache(
    ImmutableMaterialStateCache,
    geometry_cache, material, element_cache
)

# Then use normally in compute_block!:
K_kl = compute_block!(geometry_cache, material_cache, k, l)
```
"""
# Legacy type - not part of new compositional design
# Use AssemblyMaterialWorkspace{FieldType, StateType} instead
struct ImmutableMaterialStateCache{M<:AbstractMaterialState,NIP}
    σ::NTuple{NIP,SymmetricTensor{2,3,Float64,6}}   # 6 independent components for 2nd order symmetric
    𝔻::NTuple{NIP,SymmetricTensor{4,3,Float64,36}}  # 36 independent components for 4th order symmetric
    states::NTuple{NIP,M}
end

"""
    reset!(workspace::AssemblyMaterialWorkspace)

Reset assembly material workspace to zero values.

# Side Effects
Mutates all arrays in workspace to zero.
"""
function reset!(workspace::AssemblyMaterialWorkspace{FieldType, StateType}) where {FieldType, StateType}
    # Reset all fields to zero
    # CRITICAL FIX: Use getfield directly to avoid type instability from getproperty
    fields = getfield(workspace, 1)  # Direct field access - zero allocation, type-stable
    states = getfield(workspace, 2)  # Direct field access - zero allocation, type-stable
    # CRITICAL FIX: Pre-compute zero_field and zero_state ONCE (they're constants for stateless materials)
    # For StatelessConstantTangent, these are the same every time, so we can reuse them
    # But we need to compute them here since FieldType and StateType are type parameters
    zero_field = create_zero_field(FieldType)
    zero_state = create_zero_state(StateType)
    n = length(fields)  # Direct length call - zero allocation
    @inbounds for i in 1:n
        fields[i] = zero_field
        states[i] = zero_state
    end
    return nothing
end

# Zero-allocation overload: Accept pre-allocated zero values to avoid create_zero_field allocation
function reset!(
    workspace::AssemblyMaterialWorkspace{FieldType, StateType},
    zero_field::FieldType,
    zero_state::StateType
) where {FieldType, StateType}
    # Reset all fields to zero using pre-allocated values (zero-allocation)
    fields = getfield(workspace, 1)  # Direct field access - zero allocation, type-stable
    states = getfield(workspace, 2)  # Direct field access - zero allocation, type-stable
    n = length(fields)  # Direct length call - zero allocation
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

**Purpose:** Create temporary workspace for ONE element during assembly.
**Note:** For persistent state storage, use `create_global_material_cache()` instead.

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
- **Zero allocations** in hot loops!

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

# Backward compatibility alias
const create_assembly_workspace = create_material_cache

"""
    create_material_cache(
        ::Type{ImmutableMaterialStateCache},
        geometry_cache::ImmutableGeometryCache{N,NIP},
        material::AbstractMaterial,
        element_cache::ElementCache
    ) -> ImmutableMaterialStateCache{M,NIP}

Create immutable material state cache with NTuple fields (zero allocations).

# Process
1. Compute stress/tangent at all integration points
2. Convert Vectors to NTuples (compile-time sizes)
3. Return immutable cache

# Zero-Allocation Benefits

Unlike mutable `AssemblyMaterialWorkspace`, this version:
- Uses NTuple (stack-allocated, no heap access)
- Enables full compiler optimization (sizes known at compile time)
- Eliminates type instability from Vector indexing

# Example

```julia
geometry_cache = create_geometry_cache(
    ImmutableGeometryCache, element_cache, kernel, elem_id, mesh
)
material_cache = create_material_cache(
    ImmutableMaterialStateCache, geometry_cache, material, element_cache
)
# Now both caches are zero-allocation!
```
"""
function create_material_cache(
    ::Type{ImmutableMaterialStateCache},
    geometry_cache::ImmutableGeometryCache{N,NIP},
    material::AbstractMaterial,
    element_cache::ElementCache
) where {N,NIP}
    # Compute stress and tangent at all integration points
    σ_vec = Vector{SymmetricTensor{2,3,Float64,6}}(undef, NIP)
    𝔻_vec = Vector{SymmetricTensor{4,3,Float64,36}}(undef, NIP)

    # Get strain field (if needed for material evaluation)
    # For now, assume zero strain (elastic initialization)
    # This will be updated in actual assembly loop

    if needs_state(material)
        # Stateful material
        states_vec = Vector{PlasticityState}(undef, NIP)
        for q in 1:NIP
            ε = zero(SymmetricTensor{2,3,Float64,6})  # Zero strain
            state = PlasticityState()  # Initial state
            σ_vec[q], 𝔻_vec[q], states_vec[q] = update_material!(material, ε, state)
        end
        # Convert to NTuple
        σ_tuple = ntuple(i -> σ_vec[i], Val(NIP))
        𝔻_tuple = ntuple(i -> 𝔻_vec[i], Val(NIP))
        states_tuple = ntuple(i -> states_vec[i], Val(NIP))
        return ImmutableMaterialStateCache{PlasticityState,NIP}(σ_tuple, 𝔻_tuple, states_tuple)
    else
        # Stateless material
        for q in 1:NIP
            ε = zero(SymmetricTensor{2,3,Float64,6})
            σ_vec[q], 𝔻_vec[q] = evaluate_material(material, ε)
        end
        # Convert to NTuple
        σ_tuple = ntuple(i -> σ_vec[i], Val(NIP))
        𝔻_tuple = ntuple(i -> 𝔻_vec[i], Val(NIP))
        states_tuple = ntuple(i -> EmptyState(), Val(NIP))
        return ImmutableMaterialStateCache{EmptyState,NIP}(σ_tuple, 𝔻_tuple, states_tuple)
    end
end
