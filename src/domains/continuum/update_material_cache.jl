# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Material cache update functions for continuum elements.

Computes stress, tangent modulus, and internal state at integration points.
"""

using Tensors

# ============================================================================
# MATERIAL BEHAVIOR DISPATCH FUNCTIONS
# ============================================================================

"""
    update_material_cache!(
        material_cache::MaterialStateCache{M},
        geometry_cache::AbstractGeometryCache,
        material::AbstractMaterial,
        ::StatelessConstantTangent,
        element_cache::ElementCache,
        state_elem,
        Δt::Float64
    ) where M

Update material cache for constant tangent materials (e.g., linear elastic).

Computes stress and tangent once, then replicates to all integration points.
Most efficient case - single material evaluation.
"""
@inline function update_material_cache!(
    material_cache::MaterialStateCache{M},
    geometry_cache::GeometryCache,
    material::AbstractMaterial,
    ::StatelessConstantTangent,
    element_cache::ElementCache,
    state_elem,
    elem_id::Int,
    Δt::Float64
) where M<:AbstractMaterialState
    nips = length(element_cache.ips)

    # Compute once at reference configuration
    E_ref = zero(SymmetricTensor{2,3,Float64,6})
    σ_ref, 𝔻_ref, _ = compute_stress(material, E_ref, nothing, 0.0)

    # Fill all IPs with same values
    for q in 1:nips
        material_cache.σ[q] = σ_ref
        material_cache.𝔻[q] = 𝔻_ref
        material_cache.states[q] = EmptyState()
    end

    return nothing
end

"""
    update_material_cache!(
        material_cache::MaterialStateCache{M},
        geometry_cache::AbstractGeometryCache,
        material::AbstractMaterial,
        ::StatelessStrainDependent,
        element_cache::ElementCache,
        state_elem,
        Δt::Float64
    ) where M

Update material cache for strain-dependent stateless materials (e.g., hyperelastic).

Computes strain, stress, and tangent at each integration point.
No internal state tracking.
"""
@inline function update_material_cache!(
    material_cache::MaterialStateCache{M},
    geometry_cache::GeometryCache,
    material::AbstractMaterial,
    ::StatelessStrainDependent,
    element_cache::ElementCache,
    state_elem,
    elem_id::Int,
    Δt::Float64
) where M<:AbstractMaterialState
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
        σ, 𝔻, _ = compute_stress(material, E, nothing, 0.0)

        material_cache.σ[q] = σ
        material_cache.𝔻[q] = 𝔻
        material_cache.states[q] = nothing
    end

    return nothing
end

"""    
    update_material_cache!(
        material_cache::MaterialStateCache{M},
        geometry_cache::AbstractGeometryCache,
        material::AbstractMaterial,
        ::StatefulStrainDependent,
        element_cache::ElementCache,
        state_old::Union{Nothing,Matrix{<:AbstractMaterialState}},
        elem_id::Int,
        Δt::Float64
    ) where M <: AbstractMaterialState

Update material cache for stateful materials (e.g., plasticity, damage).

Computes strain, stress, tangent, and updates internal state at each integration point.
Uses old state from previous time step.
"""
@inline function update_material_cache!(
    material_cache::MaterialStateCache{M},
    geometry_cache::GeometryCache,
    material::AbstractMaterial,
    ::StatefulStrainDependent,
    element_cache::ElementCache,
    state_old::Matrix{<:AbstractMaterialState},
    elem_id::Int,
    Δt::Float64
) where M<:AbstractMaterialState
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
        σ, 𝔻, state_q_new = compute_stress(material, ε, state_q_old, Δt)

        material_cache.σ[q] = σ
        material_cache.𝔻[q] = 𝔻
        material_cache.states[q] = state_q_new
    end

    return nothing
end

# ============================================================================
# MAIN DISPATCHER
# ============================================================================

"""
    update_material_cache!(
        material_cache::MaterialStateCache{M},
        geometry_cache::AbstractGeometryCache,
        material::AbstractMaterial,
        element_cache::ElementCache,
        state_old::Union{Nothing,Matrix{M}},
        elem_id::Int,
        Δt::Float64
    ) where M

Update material state cache by computing stress, tangent, and internal state.

Dispatches to behavior-specific implementations:
- **StatelessConstantTangent**: Compute once, replicate to all IPs
- **StatelessStrainDependent**: Compute at each IP (no state)
- **StatefulStrainDependent**: Compute and update state at each IP

# Arguments
- `material_cache`: Material cache to update (parametric with state type M)
- `geometry_cache`: Geometry cache (with coordinates, gradients)
- `material`: Material model
- `element_cache`: Element cache (with displacements as Vec{3})
- `state_old`: Global material state [nips, nelems] (nothing for stateless)
- `elem_id`: Current element ID
- `Δt`: Time increment

# Side Effects
Mutates material_cache.σ, material_cache.𝔻, material_cache.states

# Zero-Allocation Guarantee
No allocations - writes to pre-allocated material_cache arrays.
"""
function update_material_cache!(
    material_cache::MaterialStateCache{M},
    geometry_cache::GeometryCache,
    material::AbstractMaterial,
    element_cache::ElementCache,
    state_old::Union{Nothing,Matrix{<:AbstractMaterialState}},
    elem_id::Int,
    Δt::Float64
) where M<:AbstractMaterialState
    # Dispatch to behavior-specific implementation
    behavior = material_behavior(material)
    update_material_cache!(
        material_cache,
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
