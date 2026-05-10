# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

"""
Pass~1 helpers for DOF-based assembly with [`ContinuumKernel`](@ref):
vertex displacement scatter and material-workspace preparation.
"""

using Tensors
using Tensors: basevec, symmetric, Tensor

using ..JuliaFEM: AbstractMaterial, compute_stress, continuum_kinematics,
    GreenLagrangeKinematics, SmallStrainKinematics

"""
    scatter_vertex_displacements_from_global!(
        u_buffer, dofs_storage, configuration, ::Type{E},
    ) -> Nothing

Scatter global displacement DOFs from `configuration` into
`u_buffer` (vertex-major `Vec{3}` per topology node) using compile-time
[`local_dof_layout`](@ref)`(E)` and the element's global DOF list in
`dofs_storage`. Only entries with `1 ≤ component ≤ 3` and valid
`entity_local` vertex indices participate.

No heap allocation in the loop.
"""
@inline function scatter_vertex_displacements_from_global!(
    u_buffer::Vector{Vec{3,Float64}},
    dofs_storage::Vector{Int},
    configuration::AbstractVector{Float64},
    ::Type{E},
) where {E<:AbstractElement}
    layout = local_dof_layout(E)
    nbuf = length(u_buffer)
    @inbounds for v in 1:nbuf
        u_buffer[v] = zero(Vec{3,Float64})
    end
    @inbounds for li in eachindex(layout)
        ent = entity_local(layout[li])
        (1 ≤ ent ≤ nbuf) || continue
        comp = component(layout[li])
        (1 ≤ comp ≤ 3) || continue
        d = dofs_storage[li]
        u_buffer[ent] = u_buffer[ent] + basevec(Vec{3,Float64}, comp) * configuration[d]
    end
    return nothing
end

"""
    _continuum_fill_workspace_stress_from_displacement!(
        material_workspace, geometry_cache, element_cache, material, Δt, empty_state,
    ) -> Nothing

Recompute `(σ, 𝔻)` at every IP from nodal displacements in `element_cache.u_buffer`
using [`continuum_kinematics`](@ref)`(material)` (small strain or Green–Lagrange).

Used when `material_behavior(material) isa StatelessConstantTangent` but a global
`configuration` is supplied so Cauchy stress tracks the current displacement while
the tangent remains the constitutive tangent returned by [`compute_stress`](@ref).

Zero-allocation in the IP loop.
"""
@inline function _continuum_fill_workspace_stress_from_displacement!(
    material_workspace::AssemblyMaterialWorkspace{FieldType, StateType},
    geometry_cache::GeometryCache,
    element_cache::ElementCache,
    material::AbstractMaterial,
    Δt::Float64,
    empty_state::NamedTuple,
) where {FieldType, StateType}
    fields_mw = getfield(material_workspace, 1)
    states_mw = getfield(material_workspace, 2)
    nips = length(element_cache.ips)
    nnodes = length(geometry_cache.X)
    kin = continuum_kinematics(material)
    I = one(Tensor{2,3,Float64,9})
    @inbounds for q in 1:nips
        strain_measure = if kin isa SmallStrainKinematics
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
            throw(ArgumentError("unknown continuum kinematics $(typeof(kin))"))
        end
        σ, 𝔻, _ = compute_stress(material, strain_measure, NamedTuple(), Δt)
        fields_mw[q] = (σ=σ, 𝔻=𝔻)
        states_mw[q] = empty_state
    end
    return nothing
end

@inline function prepare_dof_based_material_workspace!(
    k_e::ContinuumKernel,
    material_workspace::AssemblyMaterialWorkspace,
    geometry_cache::GeometryCache,
    element_cache::ElementCache,
    eid::Int,
    configuration::Union{Nothing,AbstractVector{Float64}},
    global_material_cache::Union{Nothing,GlobalMaterialCache},
    Δt::Float64,
    ::Type{E},
) where {E<:AbstractElement}
    mat = k_e.material
    beh = material_behavior(mat)
    fields_mw = getfield(material_workspace, 1)
    states_mw = getfield(material_workspace, 2)
    ips_ec = getfield(element_cache, :ips)
    nips = length(ips_ec)
    if beh isa StatelessConstantTangent
        fields_ref_e, empty_state_e = reference_fields(k_e)
        if configuration !== nothing
            scatter_vertex_displacements_from_global!(
                element_cache.u_buffer, element_cache.dofs, configuration, E,
            )
            _continuum_fill_workspace_stress_from_displacement!(
                material_workspace, geometry_cache, element_cache, mat, Δt, empty_state_e,
            )
        else
            @inbounds for q in 1:nips
                fields_mw[q] = fields_ref_e
                states_mw[q] = empty_state_e
            end
        end
    elseif beh isa StatelessStrainDependent
        if configuration !== nothing
            scatter_vertex_displacements_from_global!(
                element_cache.u_buffer, element_cache.dofs, configuration, E,
            )
        end
        update_material_cache_stateless_strain!(
            material_workspace, geometry_cache, mat, element_cache, Δt,
        )
    elseif beh isa StatefulStrainDependent
        global_material_cache === nothing && throw(ArgumentError(
            "DOF-based Pass 1: StatefulStrainDependent material requires keyword " *
            "`global_material_cache=create_global_material_cache(mat; n_ips, n_elems)`",
        ))
        if configuration !== nothing
            scatter_vertex_displacements_from_global!(
                element_cache.u_buffer, element_cache.dofs, configuration, E,
            )
        end
        update_material_cache!(
            material_workspace,
            geometry_cache,
            mat,
            element_cache,
            global_material_cache,
            eid,
            Δt,
        )
    else
        throw(ArgumentError(
            "unsupported material behavior $(typeof(beh)) for ContinuumKernel in DOF-based Pass 1",
        ))
    end
    return nothing
end
