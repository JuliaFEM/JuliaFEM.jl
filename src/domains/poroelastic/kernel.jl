# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

#=
Linear quasi-steady **Biot poroelasticity** (small strain) on a single mesh:
vertex displacement `u` (same layout as [`ContinuumKernel`](@ref)) plus a
scalar **pore pressure** `p` at vertices (same low-order layout as
[`ThermoElasticKernel`](@ref) uses for temperature).

Weak form (steady stiffness blocks; no body loads shown):

    ∫ ℂ : ε(u) : ε(v) dΩ  −  α ∫ p · div(v) dΩ  = rhs_u(v)
    − α ∫ q · div(u) dΩ  +  ∫ k ∇p · ∇q dΩ      = rhs_p(q)

with `ℂ` and small-strain `ε(u)` from [`compute_stress`](@ref) on
`mech_material`, isotropic Darcy conductivity tensor `k` from
[`HydraulicConductivity`](@ref) (`k = K I` in 3D), and **Biot coefficient**
`α ≥ 0` (often denoted `b` in geotechnical notation).

The off-diagonal sign pattern matches [`ThermoElasticKernel`](@ref) with
`β → α`: both coupling blocks use the same `-α ∫ (∂N_i/∂x_α) N_j` /
`-α ∫ N_i (∂N_j/∂x_β)` structure so the assembled `K` stays symmetric.

For transient **storage + volumetric strain rate** in the fluid mass balance,
`assemble_M!` / `apply_M!` add:

    M_pp[i,j] = storage_S · ∫ N_i N_j dΩ,
    M_pu[i,j] =  α · ∫ N_i^p (∂N_j^u/∂x_{c_j}) dΩ,
    M_up[i,j] =  α · ∫ (∂N_i^u/∂x_{c_i}) N_j^p dΩ,

with `storage_S ≥ 0` and `α ≥ 0`. The `M_pu` / `M_up` blocks are Galerkin
partners of the steady `K_pu` / `K_up` couplings (assembled `M_pu = -K_pu`,
`M_up = -K_up` for the same mesh and quadrature). Optional solid density
`ρ ≥ 0` adds a consistent **`M_uu`** on displacements (same microkernel as
[`ContinuumKernel`](@ref) with `density = ρ`; `ρ = 0` skips). Rows with
`storage_S = 0` and `α = 0` still allow a transient-only mechanical mass when
`ρ > 0`.

`operator_is_posdef` is `false`: even though this symmetric operator can
be elliptic in some parameter regimes, downstream solvers should not
assume SPD (compare mixed `u`–`p` solids).

# DOF template

```julia
S = @DOFSet{u::DOF{Displacement{3}, Vertex}, p::DOF{PorePressure, Vertex}}
kernel = BiotPoroelasticKernel(
    ContinuumFormulation{ThreeDimensional}(),
    LinearElastic(E = 3.0e10, ν = 0.25),
    HydraulicConductivity(K = 1.0e-12),
    0.8,
    0.0,
    0.0,
)
```
=#

using Tensors

using ..JuliaFEM: AbstractKernel, AbstractFormulation
using ..JuliaFEM: ContinuumFormulation, AbstractContinuumTheory
using ..JuliaFEM: AbstractMaterial, LinearElastic, HydraulicConductivity
using ..JuliaFEM: hydraulic_conductivity_tensor
using ..JuliaFEM: ThermoElasticQPBuffer
using ..JuliaFEM: AssemblyMaterialWorkspace
using ..JuliaFEM: compute_stiffness_value
import ..JuliaFEM: qpoint_buffer_eltype, update_qpoint_buffer!, evaluate_entry,
                   evaluate_mass_entry,
                   reference_fields, get_field, dofs_per_node
using ..JuliaFEM: DOFLayoutEntry, field_idx, entity_local, component


"""
    BiotPoroelasticKernel{Theory, MatM, MatF}

Coupled poroelastic kernel: `u` (3 DOFs / node) + pore pressure `p` (1 DOF /
node). See the file-level comment for the steady weak form.

# Fields
- `formulation::ContinuumFormulation{Theory}`
- `mech_material::MatM` — mechanical solid (`LinearElastic`, …)
- `flow_material::MatF` — [`HydraulicConductivity`](@ref) (Darcy mobility `K`)
- `α::Float64` — Biot–Willis coefficient (set to `0.0` to recover a block-
  diagonal elasticity + Darcy potential split for regression tests)
- `storage_S::Float64` — fluid storage coefficient on pressure (`≥ 0`);
  `0` skips the `M_pp` block; `M_pu` and `M_up` still appear when `α > 0`
- `density::Float64` — solid mass density `ρ` (`≥ 0`) on `u` for `M_uu`; `0`
  skips the displacement mass block (quasi-static default)

# Example

```julia
S = @DOFSet{u::DOF{Displacement{3}, Vertex}, p::DOF{PorePressure, Vertex}}
kernel = BiotPoroelasticKernel(
    ContinuumFormulation{ThreeDimensional}(),
    LinearElastic(E = 3.0e10, ν = 0.25),
    HydraulicConductivity(K = 1.0e-12),
    0.8,
    0.0,
    0.0,
)
```
"""
struct BiotPoroelasticKernel{Theory<:AbstractContinuumTheory,
                             MatM<:AbstractMaterial,
                             MatF<:HydraulicConductivity} <: AbstractKernel
    formulation::ContinuumFormulation{Theory}
    mech_material::MatM
    flow_material::MatF
    α::Float64
    storage_S::Float64
    density::Float64

    function BiotPoroelasticKernel(
        formulation::ContinuumFormulation{Theory},
        mech_material::MatM,
        flow_material::MatF,
        α::Float64,
        storage_S::Float64,
        density::Float64,
    ) where {Theory<:AbstractContinuumTheory,
             MatM<:AbstractMaterial,
             MatF<:HydraulicConductivity}
        α ≥ 0.0 || throw(ArgumentError("Biot coefficient α must be ≥ 0, got α = $α"))
        storage_S ≥ 0.0 ||
            throw(ArgumentError("storage_S must be ≥ 0, got storage_S = $storage_S"))
        density ≥ 0.0 ||
            throw(ArgumentError("density must be ≥ 0, got density = $density"))
        return new{Theory, MatM, MatF}(
            formulation,
            mech_material,
            flow_material,
            α,
            storage_S,
            density,
        )
    end
end

function BiotPoroelasticKernel(
    formulation::ContinuumFormulation{Theory},
    mech_material::MatM,
    flow_material::MatF,
    α::Float64,
    storage_S::Float64,
) where {Theory<:AbstractContinuumTheory,
         MatM<:AbstractMaterial,
         MatF<:HydraulicConductivity}
    return BiotPoroelasticKernel(formulation, mech_material, flow_material, α, storage_S, 0.0)
end

function BiotPoroelasticKernel(
    formulation::ContinuumFormulation{Theory},
    mech_material::MatM,
    flow_material::MatF,
    α::Float64,
) where {Theory<:AbstractContinuumTheory,
         MatM<:AbstractMaterial,
         MatF<:HydraulicConductivity}
    return BiotPoroelasticKernel(formulation, mech_material, flow_material, α, 0.0, 0.0)
end

"""Convenience constructor with `α = 0` and `storage_S = 0` (block-diagonal steady split)."""
function BiotPoroelasticKernel(
    formulation::ContinuumFormulation{Theory},
    mech_material::MatM,
    flow_material::MatF,
) where {Theory<:AbstractContinuumTheory,
         MatM<:AbstractMaterial,
         MatF<:HydraulicConductivity}
    return BiotPoroelasticKernel(formulation, mech_material, flow_material, 0.0, 0.0, 0.0)
end

@inline operator_is_posdef(::BiotPoroelasticKernel) = false

@inline dofs_per_node(::BiotPoroelasticKernel) = 4

function get_field(::K) where {K<:BiotPoroelasticKernel}
    error("$(K) is multi-field — use `local_dof_layout(E)` and " *
          "`elem.dof_indices` instead of `get_field(kernel)`.")
end

# Reuse the thermo-elastic per-IP buffer layout (elasticity tangent + rank-2 tensor).
@inline qpoint_buffer_eltype(::BiotPoroelasticKernel) = ThermoElasticQPBuffer

@inline function reference_fields(kernel::BiotPoroelasticKernel)
    ε_ref = zero(SymmetricTensor{2,3,Float64,6})
    σ_ref, 𝔻_ref, _ = compute_stress(kernel.mech_material, ε_ref, NamedTuple(), 0.0)
    q_ref = zero(Vec{3,Float64})
    k_ref = hydraulic_conductivity_tensor(kernel.flow_material)
    return ((σ = σ_ref, 𝔻 = 𝔻_ref, q = q_ref, k = k_ref), NamedTuple())
end

@inline function update_qpoint_buffer!(
    buffer::AbstractVector{ThermoElasticQPBuffer},
    workspace::AssemblyMaterialWorkspace{FieldType, StateType},
    ::BiotPoroelasticKernel,
) where {FieldType, StateType}
    fields = getfield(workspace, 1)
    @inbounds for q in eachindex(buffer)
        f = fields[q]
        buffer[q] = ThermoElasticQPBuffer(f.𝔻, f.k)
    end
    return nothing
end

"""
    evaluate_entry(kernel::BiotPoroelasticKernel, geometry_cache,
                   qp_vec, layout_i, layout_j, elem_id::Int) -> Float64

| (field_i, field_j) | block | contribution |
| ------------------ | ----- | ------------ |
| (1, 1)             | K_uu  | `Σ_q B_iα : ℂ : B_jβ · detJ_w` |
| (2, 2)             | K_pp  | `Σ_q ∇N_i · k · ∇N_j · detJ_w` |
| (1, 2)             | K_up  | `−α · Σ_q (∂N_i/∂x_α) · N_j · detJ_w` |
| (2, 1)             | K_pu  | `−α · Σ_q N_i · (∂N_j/∂x_β) · detJ_w` |
"""
@inline function evaluate_entry(
    kernel::BiotPoroelasticKernel,
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
        @inbounds for q in 1:n_ips
            ∇N_i  = geometry_cache.∇N_data[q, node_i]
            ∇N_j  = geometry_cache.∇N_data[q, node_j]
            detJw = geometry_cache.detJ_w[q]
            C     = Tensor{4,3}(qp_vec[q].C)
            K_ij += compute_stiffness_value(∇N_i, ∇N_j, C, comp_i, comp_j) * detJw
        end

    elseif fi == 2 && fj == 2
        @inbounds for q in 1:n_ips
            ∇N_i  = geometry_cache.∇N_data[q, node_i]
            ∇N_j  = geometry_cache.∇N_data[q, node_j]
            detJw = geometry_cache.detJ_w[q]
            k_q   = qp_vec[q].k
            K_ij += (∇N_i ⋅ k_q ⋅ ∇N_j) * detJw
        end

    elseif fi == 1 && fj == 2
        α = kernel.α
        @inbounds for q in 1:n_ips
            ∇N_i  = geometry_cache.∇N_data[q, node_i]
            N_j   = geometry_cache.N_data[q, node_j]
            detJw = geometry_cache.detJ_w[q]
            K_ij += -α * ∇N_i[comp_i] * N_j * detJw
        end

    else  # fi == 2 && fj == 1
        α = kernel.α
        @inbounds for q in 1:n_ips
            N_i   = geometry_cache.N_data[q, node_i]
            ∇N_j  = geometry_cache.∇N_data[q, node_j]
            detJw = geometry_cache.detJ_w[q]
            K_ij += -α * N_i * ∇N_j[comp_j] * detJw
        end
    end

    return K_ij
end

"""
    evaluate_mass_entry(::BiotPoroelasticKernel, ...)

| (field_i, field_j) | block | contribution |
| ------------------ | ----- | ------------ |
| (2, 2)             | M_pp  | `storage_S · Σ_q N_i N_j detJ_w` |
| (2, 1)             | M_pu  | `α · Σ_q N_i (∂N_j/∂x_{c_j}) detJ_w` |
| (1, 2)             | M_up  | `α · Σ_q (∂N_i/∂x_{c_i}) N_j detJ_w` |
| (1, 1)             | M_uu  | `ρ · Σ_q N_i N_j detJ_w` on matching displacement components (`ρ = 0` skips) |
| other              | —     | `0` |

`M_pu` and `M_up` match minus the steady `K_pu` and `K_up` microkernels for the
same `α` and layout. `M_uu` matches [`ContinuumKernel`](@ref) mass for the same
`ρ` and vertex layout.
"""
@inline function evaluate_mass_entry(
    kernel::BiotPoroelasticKernel,
    geometry_cache,
    qp_buffer,
    layout_i::DOFLayoutEntry,
    layout_j::DOFLayoutEntry,
)
    fi = field_idx(layout_i)
    fj = field_idx(layout_j)
    n_ips = length(geometry_cache.detJ_w)

    if fi == 1 && fj == 1
        ρ = kernel.density
        ρ == 0.0 && return 0.0
        component(layout_i) == component(layout_j) || return 0.0
        node_i = entity_local(layout_i)
        node_j = entity_local(layout_j)
        M_ij = 0.0
        @inbounds for q in 1:n_ips
            N_i   = geometry_cache.N_data[q, node_i]
            N_j   = geometry_cache.N_data[q, node_j]
            detJw = geometry_cache.detJ_w[q]
            M_ij += ρ * N_i * N_j * detJw
        end
        return M_ij
    end

    if fi == 2 && fj == 2
        S = kernel.storage_S
        S == 0.0 && return 0.0
        node_i = entity_local(layout_i)
        node_j = entity_local(layout_j)
        M_ij = 0.0
        @inbounds for q in 1:n_ips
            N_i   = geometry_cache.N_data[q, node_i]
            N_j   = geometry_cache.N_data[q, node_j]
            detJw = geometry_cache.detJ_w[q]
            M_ij += S * N_i * N_j * detJw
        end
        return M_ij
    elseif fi == 2 && fj == 1
        α = kernel.α
        α == 0.0 && return 0.0
        node_i = entity_local(layout_i)
        node_j = entity_local(layout_j)
        comp_j = component(layout_j)
        M_ij = 0.0
        @inbounds for q in 1:n_ips
            N_i   = geometry_cache.N_data[q, node_i]
            ∇N_j  = geometry_cache.∇N_data[q, node_j]
            detJw = geometry_cache.detJ_w[q]
            M_ij += α * N_i * ∇N_j[comp_j] * detJw
        end
        return M_ij
    elseif fi == 1 && fj == 2
        α = kernel.α
        α == 0.0 && return 0.0
        node_i = entity_local(layout_i)
        node_j = entity_local(layout_j)
        comp_i = component(layout_i)
        M_ij = 0.0
        @inbounds for q in 1:n_ips
            ∇N_i  = geometry_cache.∇N_data[q, node_i]
            N_j   = geometry_cache.N_data[q, node_j]
            detJw = geometry_cache.detJ_w[q]
            M_ij += α * ∇N_i[comp_i] * N_j * detJw
        end
        return M_ij
    else
        return 0.0
    end
end
