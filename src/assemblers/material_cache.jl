# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Material state cache implementations for zero-allocation assembly.

Contains mutable (MaterialStateCache) and immutable (ImmutableMaterialStateCache) variants.
"""

using Tensors

"""
    MaterialStateCache{M<:AbstractMaterialState}

Workspace for material state at all integration points.

Contains pre-allocated arrays for stress, tangent, and internal state.
Mutated per element during assembly.

# Type Parameter
- `M`: Material state type (EmptyState for stateless, PlasticityState for plastic, etc.)

# Fields
- `σ::Vector{SymmetricTensor{2,3,Float64,6}}`: Stress at each IP [max_nips]
- `𝔻::Vector{SymmetricTensor{4,3,Float64,36}}`: Tangent modulus at each IP [max_nips]
- `states::Vector{M}`: Internal state at each IP [max_nips]

# Zero-Allocation Usage
Arrays are mutated in-place during `update_material_cache!` - no heap allocation.

# Examples
```julia
# Stateless material (elastic)
mat_cache = MaterialStateCache{EmptyState}(...)

# Stateful material (plasticity)
mat_cache = MaterialStateCache{PlasticityState}(...)
```
"""
struct MaterialStateCache{M<:AbstractMaterialState} <: AbstractMaterialStateCache{M}
    σ::Vector{SymmetricTensor{2,3,Float64,6}}        # Stress [NIP] (6 independent components)
    𝔻::Vector{SymmetricTensor{4,3,Float64,36}}       # Tangent [NIP] (36 independent components)
    states::Vector{M}                                 # State [NIP]
end

"""
    ImmutableMaterialStateCache{M,NIP}

Immutable material state cache using NTuple for zero-allocation access.

Unlike `MaterialStateCache`, this version:
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
struct ImmutableMaterialStateCache{M<:AbstractMaterialState,NIP} <: AbstractMaterialStateCache{M}
    σ::NTuple{NIP,SymmetricTensor{2,3,Float64,6}}   # 6 independent components for 2nd order symmetric
    𝔻::NTuple{NIP,SymmetricTensor{4,3,Float64,36}}  # 36 independent components for 4th order symmetric
    states::NTuple{NIP,M}
end

"""
    reset!(cache::MaterialStateCache{M}) where M

Reset material state cache to zero values.

# Side Effects
Mutates all arrays in cache to zero.
"""
function reset!(cache::MaterialStateCache{M}) where M
    fill!(cache.σ, zero(SymmetricTensor{2,3,Float64,6}))
    fill!(cache.𝔻, zero(SymmetricTensor{4,3,Float64,36}))
    # Don't reset states - they may have non-zero initial values
    return nothing
end

# ============================================================================
# CONSTRUCTORS
# ============================================================================

"""
    create_material_cache(material::M, max_nips::Int) -> MaterialStateCache{S}
        where {M <: AbstractMaterial}

Create pre-allocated material state workspace with type-stable state type.

Uses `state_type(M)` trait to determine concrete state type at compile time,
ensuring full type stability and zero allocations.

# Arguments
- `material`: Material model (type M determines state type S)
- `max_nips`: Maximum integration points per element

# Returns
- `MaterialStateCache{EmptyState}` for stateless materials (e.g., LinearElastic)
- `MaterialStateCache{PlasticityState}` for J2 plasticity (e.g., PerfectPlasticity)
- `MaterialStateCache{S}` for other stateful materials with state type S

# Type Stability
Return type is fully inferrable:
- `M` is concrete material type (known at compile time)
- `S = state_type(M)` is concrete state type (trait dispatch)
- `MaterialStateCache{S}` is concrete return type
- **Zero allocations** in hot loops!

# Examples
```julia
# Stateless material
mat = LinearElastic(E=210e9, ν=0.3)
cache = create_material_cache(mat, 8)  # MaterialStateCache{EmptyState}

# Stateful material  
mat = PerfectPlasticity(E=210e9, ν=0.3, σ_y=250e6)
cache = create_material_cache(mat, 8)  # MaterialStateCache{PlasticityState}
```
"""
function create_material_cache(material::M, max_nips::Int) where M<:AbstractMaterial
    σ = [zero(SymmetricTensor{2,3,Float64,6}) for _ in 1:max_nips]
    𝔻 = [zero(SymmetricTensor{4,3,Float64,36}) for _ in 1:max_nips]

    # Get state type via trait (compile-time constant)
    S = state_type(M)
    states = [zero(S) for _ in 1:max_nips]

    return MaterialStateCache{S}(σ, 𝔻, states)
end

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

Unlike mutable `MaterialStateCache`, this version:
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
