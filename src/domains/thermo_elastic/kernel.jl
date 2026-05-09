# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

#=
Coupled thermo-elastic kernel — the third concrete kernel in the codebase
and the first multi-field one. Its sole purpose right now is to prove
that the multi-field machinery introduced for the DOF-based assembler

  * `local_dof_layout(E)` returns a `NTuple{N,DOFLayoutEntry}` whose
    `field_idx` actually varies (1 for `u`, 2 for `T`),
  * `_prepare_caches!` reads `elem.dof_indices` directly so it doesn't
    care that the kernel has more than one field,
  * `evaluate_entry(kernel, geom, qp, layout_i, layout_j, elem_id)` can dispatch
    on `(field_idx(layout_i), field_idx(layout_j))` to compute the
    correct stiffness block,

flows end-to-end through both `assemble!` and `apply_K!` (CPU + KA) with
zero allocations and matrix-free correctness.

Coupling form
=============

Standard small-strain thermo-elasticity. Strain ε is split additively
into mechanical and thermal parts, ε = ε_m + α·ΔT·I, and the residual
linearisation gives the symmetric coupled tangent

    K_uu[i,α; j,β]  =  ∫ B_iα : ℂ : B_jβ                          (elasticity)
    K_TT[i_T; j_T]  =  ∫ ∇N_i · k · ∇N_j                          (heat)
    K_uT[i,α; j_T]  = -β · ∫ (∂N_i/∂x_α) · N_j  dV                (mechanical-from-thermal)
    K_Tu[i_T; j,β]  = -β · ∫ N_i · (∂N_j/∂x_β) dV   (= K_uT^T)    (thermal-from-mechanical)

`β` is the kernel-level coupling coefficient (collects ℂ:αI; isotropic).
Setting `β = 0` recovers a pure block-diagonal `[K_uu 0; 0 K_TT]`
(useful as a smoke test).

Both gradient values `∇N` and basis values `N` are read from the SoA
geometry batches via `GeometryCache.∇N_data` / `GeometryCache.N_data`
(the latter is what was added in the D-next refactor).
=#

using Tensors

using ..JuliaFEM: AbstractKernel, AbstractFormulation
using ..JuliaFEM: ContinuumFormulation, FullThreeD, AbstractContinuumTheory
using ..JuliaFEM: AbstractMaterial, LinearElastic, HeatConductivity
using ..JuliaFEM: elasticity_tensor, conductivity_tensor
using ..JuliaFEM: AssemblyMaterialWorkspace
using ..JuliaFEM: compute_stiffness_value
import ..JuliaFEM: qpoint_buffer_eltype, update_qpoint_buffer!, evaluate_entry,
                   reference_fields, get_field, dofs_per_node
using ..JuliaFEM: DOFLayoutEntry, field_idx, entity_local, component


"""
    ThermoElasticQPBuffer

Per-quadrature-point buffer for `ThermoElasticKernel`. Carries the full
elasticity tangent `C` *and* the conductivity tensor `k` together so a
single column view `view(qp_buffers, :, eid)` covers both blocks.

Both materials in the current implementation are stateless, so the
values are kernel-level constants in practice — but the buffer-per-IP
shape is preserved so a future stress-dependent or temperature-dependent
material drops in without changing the assembler.

Allocation-free: `isbitstype(ThermoElasticQPBuffer) == true`.
"""
struct ThermoElasticQPBuffer
    C::SymmetricTensor{4,3,Float64,36}
    k::SymmetricTensor{2,3,Float64,6}
end


"""
    ThermoElasticKernel{Theory, MatM, MatT}

Coupled thermo-elastic kernel: vector displacement `u` (3 DOFs/node) +
scalar temperature `T` (1 DOF/node), so 4 DOFs per node. This is the
first multi-field kernel that flows through the DOF-based assembler.

# Fields
- `formulation::ContinuumFormulation{Theory}`
- `mech_material::MatM` — elastic material (`LinearElastic` for now)
- `therm_material::MatT` — heat-conductivity material
- `β::Float64` — thermo-elastic coupling coefficient (set to `0.0` for
  a block-diagonal smoke-test setup; non-zero exercises off-diagonal
  blocks)

See the file-level comment for the (deliberately simplified, gradient-
only) coupling form actually evaluated in `evaluate_entry`.
"""
struct ThermoElasticKernel{Theory<:AbstractContinuumTheory,
                           MatM<:AbstractMaterial,
                           MatT<:HeatConductivity} <: AbstractKernel
    formulation::ContinuumFormulation{Theory}
    mech_material::MatM
    therm_material::MatT
    β::Float64
end

"Convenience constructor with `β = 0` (block-diagonal smoke-test setup)."
function ThermoElasticKernel(formulation::ContinuumFormulation{Theory},
                             mech_material::MatM,
                             therm_material::MatT) where {
        Theory<:AbstractContinuumTheory,
        MatM<:AbstractMaterial,
        MatT<:HeatConductivity}
    return ThermoElasticKernel(formulation, mech_material, therm_material, 0.0)
end


# ----------------------------------------------------------------------------
# Field interface — multi-field, so single-field methods don't apply
# ----------------------------------------------------------------------------

# 3 displacement components + 1 temperature = 4 DOFs/node. Determines
# the `element_cache.dofs` allocation size in `create_element_cache`;
# the actual per-element DOF layout still comes from the element
# template's `local_dof_layout`.
@inline dofs_per_node(::ThermoElasticKernel) = 4

# Multi-field kernels deliberately do not implement `get_field`. Anyone
# calling it is on the legacy single-field path and should switch to the
# element-template DOF layout (`local_dof_layout(E)` /
# `elem.dof_indices`).
function get_field(::K) where {K<:ThermoElasticKernel}
    error("$(K) is multi-field — use `local_dof_layout(E)` and " *
          "`elem.dof_indices` instead of `get_field(kernel)`.")
end

# ----------------------------------------------------------------------------
# Microkernel contract
# ----------------------------------------------------------------------------

@inline qpoint_buffer_eltype(::ThermoElasticKernel) = ThermoElasticQPBuffer


"""
    reference_fields(kernel::ThermoElasticKernel)

Per-IP reference state: `(σ, 𝔻, q, k)`. The first two come from the
mechanical material at zero strain, the last two from the thermal
material at zero gradient — both are stateless, so the values are
constants reused at every IP by `_prepare_caches!`.
"""
@inline function reference_fields(kernel::ThermoElasticKernel)
    σ_ref = zero(SymmetricTensor{2,3,Float64,6})
    𝔻_ref = elasticity_tensor(kernel.mech_material)
    q_ref = zero(Vec{3,Float64})
    k_ref = conductivity_tensor(kernel.therm_material)
    return ((σ = σ_ref, 𝔻 = 𝔻_ref, q = q_ref, k = k_ref), NamedTuple())
end


"""
    update_qpoint_buffer!(buffer, workspace, ::ThermoElasticKernel)

Pack `(𝔻, k)` from the per-element material workspace into the per-IP
`ThermoElasticQPBuffer` the assembler keeps in `qp_buffers`.
Allocation-free.
"""
@inline function update_qpoint_buffer!(
    buffer::AbstractVector{ThermoElasticQPBuffer},
    workspace::AssemblyMaterialWorkspace{FieldType, StateType},
    ::ThermoElasticKernel,
) where {FieldType, StateType}
    fields = getfield(workspace, 1)
    @inbounds for q in eachindex(buffer)
        f = fields[q]
        buffer[q] = ThermoElasticQPBuffer(f.𝔻, f.k)
    end
    return nothing
end


"""
    evaluate_entry(kernel::ThermoElasticKernel, geometry_cache,
                   qp_vec::AbstractVector{ThermoElasticQPBuffer},
                   layout_i::DOFLayoutEntry, layout_j::DOFLayoutEntry,
                   elem_id::Int) -> Float64

Multi-field DOF-based microkernel. Dispatches on the (field_i, field_j)
pair. The volume kernel ignores `elem_id`.

| (field_i, field_j) | block  | formula                                              |
| ------------------ | ------ | ---------------------------------------------------- |
| (1, 1)             | K_uu   | `Σ_q B_iα : ℂ : B_jβ · detJ_w`                       |
| (2, 2)             | K_TT   | `Σ_q ∇N_i · k · ∇N_j · detJ_w`                       |
| (1, 2)             | K_uT   | `-β · Σ_q (∂N_i/∂x_α) · N_j · detJ_w`                |
| (2, 1)             | K_Tu   | `-β · Σ_q N_i · (∂N_j/∂x_β) · detJ_w` (= K_uT^T)     |

K_uT/K_Tu now use the standard ε-T form (`∇N_i · N_j`) made possible by
the SoA `N_data` batch added to `GeometryCache`.

Allocation-free; reads both `∇N_data` and `N_data` views.
"""
@inline function evaluate_entry(
    kernel::ThermoElasticKernel,
    geometry_cache,
    qp_vec::AbstractVector{ThermoElasticQPBuffer},
    layout_i::DOFLayoutEntry,
    layout_j::DOFLayoutEntry,
    ::Int,
)
    fi = field_idx(layout_i)
    fj = field_idx(layout_j)

    node_i = entity_local(layout_i)
    node_j = entity_local(layout_j)
    comp_i = component(layout_i)
    comp_j = component(layout_j)

    n_ips = length(geometry_cache.detJ_w)
    K_ij  = 0.0

    if fi == 1 && fj == 1
        # K_uu — standard elasticity
        @inbounds for q in 1:n_ips
            ∇N_i  = geometry_cache.∇N_data[q, node_i]
            ∇N_j  = geometry_cache.∇N_data[q, node_j]
            detJw = geometry_cache.detJ_w[q]
            C     = Tensor{4,3}(qp_vec[q].C)
            K_ij += compute_stiffness_value(∇N_i, ∇N_j, C, comp_i, comp_j) * detJw
        end

    elseif fi == 2 && fj == 2
        # K_TT — standard heat conduction
        @inbounds for q in 1:n_ips
            ∇N_i  = geometry_cache.∇N_data[q, node_i]
            ∇N_j  = geometry_cache.∇N_data[q, node_j]
            detJw = geometry_cache.detJ_w[q]
            k_q   = qp_vec[q].k
            K_ij += (∇N_i ⋅ k_q ⋅ ∇N_j) * detJw
        end

    elseif fi == 1 && fj == 2
        # K_uT — standard ε-T coupling: -β · ∫ (∂N_i/∂x_α) · N_j  dV
        β = kernel.β
        @inbounds for q in 1:n_ips
            ∇N_i  = geometry_cache.∇N_data[q, node_i]
            N_j   = geometry_cache.N_data[q, node_j]
            detJw = geometry_cache.detJ_w[q]
            K_ij += -β * ∇N_i[comp_i] * N_j * detJw
        end

    else  # fi == 2 && fj == 1  →  K_Tu = K_uT^T
        # Symmetric counterpart: -β · ∫ N_i · (∂N_j/∂x_β) dV
        β = kernel.β
        @inbounds for q in 1:n_ips
            N_i   = geometry_cache.N_data[q, node_i]
            ∇N_j  = geometry_cache.∇N_data[q, node_j]
            detJw = geometry_cache.detJ_w[q]
            K_ij += -β * N_i * ∇N_j[comp_j] * detJw
        end
    end

    return K_ij
end
