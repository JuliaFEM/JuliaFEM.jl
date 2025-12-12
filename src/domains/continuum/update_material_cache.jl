# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Material cache update functions for continuum elements.

Computes stress, tangent modulus, and internal state at integration points.
"""

using Tensors
using ..JuliaFEM: GlobalMaterialCache, get_old_state, set_state!, material_state_type, create_zero_state

# ============================================================================
# MATERIAL BEHAVIOR DISPATCH FUNCTIONS
# ============================================================================

"""
    update_material_cache!(
        material_workspace::AssemblyMaterialWorkspace{M},
        geometry_cache::AbstractGeometryCache,
        material::AbstractMaterial,
        ::StatelessConstantTangent,
        element_cache::ElementCache,
        state_elem,
        Δt::Float64
    ) where M

Update assembly material workspace for constant tangent materials (e.g., linear elastic).

Computes stress and tangent once, then replicates to all integration points.
Most efficient case - single material evaluation.
"""
# Both mechanics and general structs use same implementation
@inline function update_material_cache!(
    material_workspace::AssemblyMaterialWorkspace,
    geometry_cache::GeometryCache,
    material::AbstractMaterial,
    ::StatelessConstantTangent,
    element_cache::ElementCache,
    state_elem,
    elem_id::Int,
    Δt::Float64
)
    nips = length(element_cache.ips)

    # Compute once at reference configuration
    E_ref = zero(SymmetricTensor{2,3,Float64,6})
    σ_ref, 𝔻_ref, _ = compute_stress(material, E_ref, NamedTuple(), 0.0)

    # Fill all IPs with same values
    # Pre-create NamedTuple once for zero allocation
    fields_ref = (σ=σ_ref, 𝔻=𝔻_ref)
    @inbounds for q in 1:nips
        material_workspace.fields[q] = fields_ref  # Direct assignment - zero allocation
        material_workspace.states[q] = NamedTuple()
    end

    return nothing
end

"""
    update_material_cache!(
        material_workspace::AssemblyMaterialWorkspace{M},
        geometry_cache::AbstractGeometryCache,
        material::AbstractMaterial,
        ::StatelessStrainDependent,
        element_cache::ElementCache,
        state_elem,
        Δt::Float64
    ) where M

Update assembly material workspace for strain-dependent stateless materials (e.g., hyperelastic).

Computes strain, stress, and tangent at each integration point.
No internal state tracking.
"""
# Both mechanics and general structs use same implementation
@inline function update_material_cache!(
    material_workspace::AssemblyMaterialWorkspace,
    geometry_cache::GeometryCache,
    material::AbstractMaterial,
    ::StatelessStrainDependent,
    element_cache::ElementCache,
    state_elem,
    elem_id::Int,
    Δt::Float64
)
    nips = length(element_cache.ips)
    nnodes = length(geometry_cache.X)
    I = one(Tensor{2,3,Float64,9})

    # Compute at each integration point
    @inbounds for q in 1:nips
        # Deformation gradient: F = I + ∇u
        F = I
        for k in 1:nnodes
            u_k = element_cache.u_buffer[k]
            ∇N_k_q = geometry_cache.∇N_data[q, k]
            F += u_k ⊗ ∇N_k_q
        end

        # Green-Lagrange strain: E = ½(C - I) = ½(F'F - I)
        C_tensor = symmetric(F' ⋅ F)
        E = SymmetricTensor{2,3}(0.5 * (C_tensor - I))

        # Compute stress and tangent
        σ, 𝔻, _ = compute_stress(material, E, NamedTuple(), 0.0)

        # Direct assignment - NamedTuple creation is zero allocation (Julia reuses instances)
        @inbounds material_workspace.fields[q] = (σ=σ, 𝔻=𝔻)
        @inbounds material_workspace.states[q] = NamedTuple()
    end

    return nothing
end

"""    
    update_material_cache!(
        material_workspace::AssemblyMaterialWorkspace{M},
        geometry_cache::AbstractGeometryCache,
        material::AbstractMaterial,
        ::StatefulStrainDependent,
        element_cache::ElementCache,
        state_old::Union{Nothing,Matrix{<:AbstractMaterialState}},
        elem_id::Int,
        Δt::Float64
    ) where M <: AbstractMaterialState

Update assembly material workspace for stateful materials (e.g., plasticity, damage).

Computes strain, stress, tangent, and updates internal state at each integration point.
Uses old state from previous time step.
"""
@inline function update_material_cache!(
    material_workspace::AssemblyMaterialWorkspace,
    geometry_cache::GeometryCache,
    material::AbstractMaterial,
    ::StatefulStrainDependent,
    element_cache::ElementCache,
    state_old::Matrix{<:AbstractMaterialState},
    elem_id::Int,
    Δt::Float64
)
    nips = length(element_cache.ips)
    nnodes = length(geometry_cache.X)

    # Extract state_old for this element
    state_elem = @view state_old[:, elem_id]

    # Compute and update state at each integration point
    @inbounds for q in 1:nips
        # Small strain: ε = sym(∇u)
        ε = zero(SymmetricTensor{2,3,Float64,6})
        for k in 1:nnodes
            u_k = element_cache.u_buffer[k]
            ∇N_k_q = geometry_cache.∇N_data[q, k]
            ε += symmetric(u_k ⊗ ∇N_k_q)
        end

        # Get old state at this IP
        state_q_old = state_elem[q]

        # Compute stress, tangent, and updated state
        # Legacy API: state_old might be AbstractMaterialState (monolithic) or NamedTuple
        # compute_stress now expects NamedTuple, so convert if needed
        if state_q_old isa NamedTuple
            state_old_nt = state_q_old
        else
            # Legacy monolithic state - convert to NamedTuple
            # For now, pass as-is and let compute_stress handle it (it accepts both)
            state_old_nt = state_q_old
        end
        σ, 𝔻, state_q_new = compute_stress(material, ε, state_old_nt, Δt)
        
        # Convert state_new to NamedTuple if it's still monolithic
        if state_q_new isa NamedTuple
            state_new_nt = state_q_new
        else
            # Legacy: convert monolithic state to NamedTuple
            # This shouldn't happen with new materials, but handle for backward compatibility
            StateType = material_state_type(material)
            state_new_nt = create_zero_state(StateType)  # Fallback to zero state
        end

        # Direct assignment - NamedTuple creation is zero allocation (Julia reuses instances)
        @inbounds material_workspace.fields[q] = (σ=σ, 𝔻=𝔻)
        material_workspace.states[q] = state_q_new
    end

    return nothing
end

# ============================================================================
# GLOBAL MATERIAL CACHE OVERLOADS (NEW API)
# ============================================================================

"""
    update_material_cache!(
        material_workspace::AssemblyMaterialWorkspace{M},
        geometry_cache::AbstractGeometryCache,
        material::AbstractMaterial,
        element_cache::ElementCache,
        global_cache::GlobalMaterialCache,
        elem_id::Int,
        Δt::Float64
    )

Update assembly material workspace using GlobalMaterialCache for state storage.

**New API:** Uses `GlobalMaterialCache` for persistent state storage.
Reads old state from `global_cache` and writes new state back.

# Arguments
- `material_workspace`: Assembly material workspace to update (temporary, per-element)
- `geometry_cache`: Geometry cache (with coordinates, gradients)
- `material`: Material model
- `element_cache`: Element cache (with displacements as Vec{3})
- `global_cache`: Global material cache (persistent state storage)
- `elem_id`: Current element ID
- `Δt`: Time increment

# Side Effects
- Mutates `material_workspace.σ`, `material_workspace.𝔻`, `material_workspace.states`
- Writes new state to `global_cache` via `set_state!()`

# Zero-Allocation Guarantee
No allocations - reads/writes to pre-allocated caches.
"""
# Use multiple dispatch instead of isa checks for type stability
function update_material_cache!(
    material_workspace::AssemblyMaterialWorkspace,
    geometry_cache::GeometryCache,
    material::AbstractMaterial,
    element_cache::ElementCache,
    global_cache::GlobalMaterialCache,
    elem_id::Int,
    Δt::Float64
)
    behavior = material_behavior(material)
    return update_material_cache!(
        material_workspace,
        geometry_cache,
        material,
        behavior,
        element_cache,
        global_cache,
        elem_id,
        Δt
    )
end

# StatelessConstantTangent - dispatch on behavior type
# Overload that accepts pre-allocated NamedTuples (zero-allocation)
# CRITICAL FIX: Add type parameters to material_workspace for type stability
@inline function update_material_cache!(
    material_workspace::AssemblyMaterialWorkspace{FieldType, StateType},
    geometry_cache::GeometryCache,
    material::AbstractMaterial,
    behavior::StatelessConstantTangent,
    element_cache::ElementCache,
    global_cache::GlobalMaterialCache,
    elem_id::Int,
    Δt::Float64,
    fields_ref::NamedTuple,  # Pre-allocated NamedTuple (from cache)
    empty_state::NamedTuple  # Pre-allocated empty state (from cache)
) where {FieldType, StateType}
    # CRITICAL FIX: Use getfield directly to avoid type instability from getproperty
    # getproperty creates Val(name) at runtime, causing type instability and allocations
    # Direct getfield access is type-stable and zero-allocation
    fields = getfield(material_workspace, 1)  # Direct field access - zero allocation, type-stable
    states = getfield(material_workspace, 2)  # Direct field access - zero allocation, type-stable
    ips = getfield(element_cache, :ips)  # Direct field access
    nips = length(ips)  # Use cached reference
    
    # Use pre-allocated NamedTuples (zero-allocation)
    @inbounds for q in 1:nips
        fields[q] = fields_ref  # Reuse pre-allocated NT - zero allocation
        states[q] = empty_state  # Reuse pre-allocated empty state
    end

    return nothing
end

# Fallback: compute NamedTuples if not provided (for backward compatibility)
@inline function update_material_cache!(
    material_workspace::AssemblyMaterialWorkspace,
    geometry_cache::GeometryCache,
    material::AbstractMaterial,
    ::StatelessConstantTangent,
    element_cache::ElementCache,
    global_cache::GlobalMaterialCache,
    elem_id::Int,
    Δt::Float64
)
    nips = length(element_cache.ips)

    # Compute once at reference configuration
    E_ref = zero(SymmetricTensor{2,3,Float64,6})
    σ_ref, 𝔻_ref, _ = compute_stress(material, E_ref, NamedTuple(), 0.0)
    
    # Create NamedTuple (allocates, but only for backward compatibility)
    fields_ref = (σ=σ_ref, 𝔻=𝔻_ref)
    empty_state = NamedTuple()
    @inbounds for q in 1:nips
        material_workspace.fields[q] = fields_ref
        material_workspace.states[q] = empty_state
    end

    return nothing
end

# StatelessStrainDependent - dispatch on behavior type
@inline function update_material_cache!(
    material_workspace::AssemblyMaterialWorkspace,
    geometry_cache::GeometryCache,
    material::AbstractMaterial,
    ::StatelessStrainDependent,
    element_cache::ElementCache,
    global_cache::GlobalMaterialCache,
    elem_id::Int,
    Δt::Float64
)
    nips = length(element_cache.ips)
    nnodes = length(geometry_cache.X)
    I = one(Tensor{2,3,Float64,9})

    # Compute at each integration point
    @inbounds for q in 1:nips
        # Deformation gradient: F = I + ∇u
        F = I
        for k in 1:nnodes
            u_k = element_cache.u_buffer[k]
            ∇N_k_q = geometry_cache.∇N_data[q, k]
            F += u_k ⊗ ∇N_k_q
        end

        # Green-Lagrange strain: E = ½(C - I) = ½(F'F - I)
        C_tensor = symmetric(F' ⋅ F)
        E = SymmetricTensor{2,3}(0.5 * (C_tensor - I))

        # Compute stress and tangent
        σ, 𝔻, _ = compute_stress(material, E, NamedTuple(), 0.0)
        
        # Direct assignment - NamedTuple creation is zero allocation (Julia reuses instances)
        @inbounds material_workspace.fields[q] = (σ=σ, 𝔻=𝔻)
        material_workspace.states[q] = NamedTuple()
    end

    return nothing
end

# StatefulStrainDependent - dispatch on behavior type
@inline function update_material_cache!(
    material_workspace::AssemblyMaterialWorkspace,
    geometry_cache::GeometryCache,
    material::AbstractMaterial,
    ::StatefulStrainDependent,
    element_cache::ElementCache,
    global_cache::GlobalMaterialCache,
    elem_id::Int,
    Δt::Float64
)
    nips = length(element_cache.ips)
    nnodes = length(geometry_cache.X)

    # Stateful - read from global_cache, compute, write back
    @inbounds for q in 1:nips
        # Small strain: ε = sym(∇u)
        ε = zero(SymmetricTensor{2,3,Float64,6})
        for k in 1:nnodes
            u_k = element_cache.u_buffer[k]
            ∇N_k_q = geometry_cache.∇N_data[q, k]
            ε += symmetric(u_k ⊗ ∇N_k_q)
        end

        # Get old state from global cache
        state_old = get_old_state(global_cache, q, elem_id)
        
        # Compute stress, tangent, and updated state
        σ, 𝔻, state_new = compute_stress(material, ε, state_old, Δt)
        
        # Direct assignment - NamedTuple creation is zero allocation (Julia reuses instances)
        @inbounds material_workspace.fields[q] = (σ=σ, 𝔻=𝔻)
        
        # Write new state back to global cache
        set_state!(global_cache, q, elem_id, state_new)
    end

    return nothing
end

# ============================================================================
# MAIN DISPATCHER (LEGACY API - Matrix{<:AbstractMaterialState})
# ============================================================================

"""
    update_material_cache!(
        material_workspace::AssemblyMaterialWorkspace,
        geometry_cache::AbstractGeometryCache,
        material::AbstractMaterial,
        element_cache::ElementCache,
        state_old::Union{Nothing,Matrix{<:AbstractMaterialState}},
        elem_id::Int,
        Δt::Float64
    )

Update assembly material workspace by computing stress, tangent, and internal state.

**Legacy API:** Uses `Matrix{<:AbstractMaterialState}` for state storage.
For new code, prefer `GlobalMaterialCache` overload.

Dispatches to behavior-specific implementations:
- **StatelessConstantTangent**: Compute once, replicate to all IPs
- **StatelessStrainDependent**: Compute at each IP (no state)
- **StatefulStrainDependent**: Compute and update state at each IP

# Arguments
- `material_workspace`: Assembly material workspace to update
- `geometry_cache`: Geometry cache (with coordinates, gradients)
- `material`: Material model
- `element_cache`: Element cache (with displacements as Vec{3})
- `state_old`: Global material state [nips, nelems] (nothing for stateless)
- `elem_id`: Current element ID
- `Δt`: Time increment

# Side Effects
Mutates material_workspace.fields and material_workspace.states

# Zero-Allocation Guarantee
No allocations - writes to pre-allocated material_workspace arrays.
"""
function update_material_cache!(
    material_workspace::AssemblyMaterialWorkspace,
    geometry_cache::GeometryCache,
    material::AbstractMaterial,
    element_cache::ElementCache,
    state_old::Union{Nothing,Matrix{<:AbstractMaterialState}},
    elem_id::Int,
    Δt::Float64
)
    # Dispatch to behavior-specific implementation
    behavior = material_behavior(material)
    update_material_cache!(
        material_workspace,
        geometry_cache,
        material,
        behavior,
        element_cache,
        state_old,
        elem_id,
        Δt
    )

    return nothing
end
