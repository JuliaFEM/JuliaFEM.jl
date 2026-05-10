# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

"""
Material cache update functions for continuum elements.

Computes stress, tangent modulus, and internal state at integration points.
"""

using Tensors
using ..JuliaFEM: GlobalMaterialCache, get_old_state, set_state!
using ..JuliaFEM: continuum_kinematics, SmallStrainKinematics, GreenLagrangeKinematics
using ..JuliaFEM: material_behavior, StatelessStrainDependent

# ============================================================================
# GLOBAL MATERIAL CACHE — behavior-dispatched implementations
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

Update assembly material workspace using `GlobalMaterialCache` for state storage.

Reads the old state from `global_cache`, computes stress and tangent, and
writes the new state back. Behavior dispatches on `material_behavior(material)`
so stateless / strain-dependent / stateful materials all share this entry.

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
function update_material_cache!(
    material_workspace::AssemblyMaterialWorkspace,
    geometry_cache::GeometryCache,
    material::AbstractMaterial,
    element_cache::ElementCache,
    global_cache::GlobalMaterialCache,
    elem_id::Int,
    Δt::Float64,
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
        Δt,
    )
end

# StatelessConstantTangent — compute stress and tangent once at the reference
# configuration and replicate to every IP. Both NamedTuples are pre-allocated
# in the cache, so the per-IP loop is zero-allocation.
@inline function update_material_cache!(
    material_workspace::AssemblyMaterialWorkspace{FieldType, StateType},
    geometry_cache::GeometryCache,
    material::AbstractMaterial,
    behavior::StatelessConstantTangent,
    element_cache::ElementCache,
    global_cache::GlobalMaterialCache,
    elem_id::Int,
    Δt::Float64,
    fields_ref::NamedTuple,
    empty_state::NamedTuple,
) where {FieldType, StateType}
    # `getfield` is type-stable; `getproperty` would allocate here.
    fields = getfield(material_workspace, 1)
    states = getfield(material_workspace, 2)
    ips = getfield(element_cache, :ips)
    nips = length(ips)

    @inbounds for q in 1:nips
        fields[q] = fields_ref
        states[q] = empty_state
    end

    return nothing
end

# Convenience overload that builds the NamedTuples on demand. Used by the
# behavior dispatcher when no pre-allocated tuples are passed in.
@inline function update_material_cache!(
    material_workspace::AssemblyMaterialWorkspace,
    geometry_cache::GeometryCache,
    material::AbstractMaterial,
    ::StatelessConstantTangent,
    element_cache::ElementCache,
    global_cache::GlobalMaterialCache,
    elem_id::Int,
    Δt::Float64,
)
    nips = length(element_cache.ips)

    E_ref = zero(SymmetricTensor{2,3,Float64,6})
    σ_ref, 𝔻_ref, _ = compute_stress(material, E_ref, NamedTuple(), 0.0)

    fields_ref = (σ=σ_ref, 𝔻=𝔻_ref)
    empty_state = NamedTuple()
    @inbounds for q in 1:nips
        material_workspace.fields[q] = fields_ref
        material_workspace.states[q] = empty_state
    end

    return nothing
end

"""
    update_material_cache_stateless_strain!(
        material_workspace, geometry_cache, material, element_cache, Δt,
    ) -> Nothing

Finite-strain / hyperelastic branch with **no** persistent integration-point
state in [`GlobalMaterialCache`](@ref). Requires
`material_behavior(material) isa StatelessStrainDependent`.

`element_cache.u_buffer` must hold the current nodal displacements (vertex
ordering matching `geometry_cache.∇N_data`). Used by the DOF-based Pass 1
when the configuration vector is supplied (or zero displacement when it
is not).

Zero-allocation in the integration loop.
"""
@inline function update_material_cache_stateless_strain!(
    material_workspace::AssemblyMaterialWorkspace,
    geometry_cache::GeometryCache,
    material::AbstractMaterial,
    element_cache::ElementCache,
    Δt::Float64,
)
    material_behavior(material) isa StatelessStrainDependent ||
        throw(ArgumentError("update_material_cache_stateless_strain! requires StatelessStrainDependent material"))
    nips = length(element_cache.ips)
    nnodes = length(geometry_cache.X)
    I = one(Tensor{2,3,Float64,9})

    @inbounds for q in 1:nips
        F = I
        for k in 1:nnodes
            u_k = element_cache.u_buffer[k]
            ∇N_k_q = geometry_cache.∇N_data[q, k]
            F += u_k ⊗ ∇N_k_q
        end

        C_tensor = symmetric(F' ⋅ F)
        E = SymmetricTensor{2,3}(0.5 * (C_tensor - I))

        σ, 𝔻, _ = compute_stress(material, E, NamedTuple(), Δt)

        @inbounds material_workspace.fields[q] = (σ=σ, 𝔻=𝔻)
        material_workspace.states[q] = NamedTuple()
    end

    return nothing
end

# StatelessStrainDependent — strain at each IP, no persistent state.
@inline function update_material_cache!(
    material_workspace::AssemblyMaterialWorkspace,
    geometry_cache::GeometryCache,
    material::AbstractMaterial,
    ::StatelessStrainDependent,
    element_cache::ElementCache,
    global_cache::GlobalMaterialCache,
    elem_id::Int,
    Δt::Float64,
)
    return update_material_cache_stateless_strain!(
        material_workspace, geometry_cache, material, element_cache, Δt,
    )
end

# StatefulStrainDependent — read old state from `global_cache`, compute the
# stress / tangent / new state at each IP, and write the new state back.
@inline function update_material_cache!(
    material_workspace::AssemblyMaterialWorkspace,
    geometry_cache::GeometryCache,
    material::AbstractMaterial,
    ::StatefulStrainDependent,
    element_cache::ElementCache,
    global_cache::GlobalMaterialCache,
    elem_id::Int,
    Δt::Float64,
)
    nips = length(element_cache.ips)
    nnodes = length(geometry_cache.X)
    kin = continuum_kinematics(material)
    I = one(Tensor{2,3,Float64,9})

    @inbounds for q in 1:nips
        ε_gl_or_small = if kin isa SmallStrainKinematics
            ε = zero(SymmetricTensor{2,3,Float64,6})
            for k in 1:nnodes
                u_k = element_cache.u_buffer[k]
                ∇N_k_q = geometry_cache.∇N_data[q, k]
                ε += symmetric(u_k ⊗ ∇N_k_q)
            end
            ε
        elseif kin isa GreenLagrangeKinematics
            F = I
            for k in 1:nnodes
                u_k = element_cache.u_buffer[k]
                ∇N_k_q = geometry_cache.∇N_data[q, k]
                F += u_k ⊗ ∇N_k_q
            end
            C_tensor = symmetric(F' ⋅ F)
            SymmetricTensor{2,3}(0.5 * (C_tensor - I))
        else
            error("unknown continuum kinematics $kin")
        end

        state_old = get_old_state(global_cache, q, elem_id)
        σ, 𝔻, state_new = compute_stress(material, ε_gl_or_small, state_old, Δt)

        @inbounds material_workspace.fields[q] = (σ=σ, 𝔻=𝔻)
        set_state!(global_cache, q, elem_id, state_new)
    end

    return nothing
end

