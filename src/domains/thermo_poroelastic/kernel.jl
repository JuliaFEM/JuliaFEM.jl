# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

#=
Linear quasi-steady **thermo-poroelasticity** (small strain) on one mesh:
vertex displacement `u`, vertex temperature `T`, and vertex pore pressure `p`.

This is the **superposition** of [`ThermoElasticKernel`](@ref) on `(u, T)` and
[`BiotPoroelasticKernel`](@ref) on `(u, p)` with the same field ordering

    field 1 → `u`,  field 2 → `T`,  field 3 → `p`.

Optional **direct `T`–`p` coupling** (defaults `0`, recovering the pure
superposition of thermo-elastic and Biot blocks):

- **`kappa_tp ≥ 0`** — steady symmetric gradient cross-coupling between `T`
  and `p` (thermo-osmosis / isotropic cross-diffusion):

      κ_tp ∫ ∇N_i · ∇N_j dΩ   for local DOF pairs in fields `(T, p)` and `(p, T)`.

  Diagonal blocks `K_TT` and `K_pp` still use only `k_T` and `k_p`.

- **`zeta_tp ≥ 0`** — symmetric L² mass between temperature and pressure for
  transient thermal pressurisation in the fluid mass balance (and the
  transpose in the energy balance if one uses the same test functions):

      ζ_tp ∫ N_T N_p dΩ  on `(T, p)` and `(p, T)` in `evaluate_mass_entry`.

`storage_S > 0` still adds L² mass on **pressure** (`M_pp`), as for
[`BiotPoroelasticKernel`](@ref). When **`α > 0`**, the kernel adds **`M_pu`**
(`p` row × `u` column) and **`M_up`** (`u` row × `p` column), each the negative
of the corresponding steady **`K_*u`** / **`K_u*`** block. When **`β > 0`**, it
adds **`M_Tu`** and **`M_uT`** on `(T,u)` and `(u,T)` as the negatives of
**`K_Tu`** and **`K_uT`**.

- **`heat_capacity ≥ 0`** — volumetric heat capacity ``ρ c_p`` on temperature
  (field 2) only, same convention as [`HeatKernel`](@ref):

      M_TT[i,j] = (ρ c_p) · ∫ N_i N_j dΩ   in `evaluate_mass_entry`.

  Default `0` skips the thermal capacity block (steady thermal energy omitted).

- **`density ≥ 0`** — solid mass density on **displacement** (field 1) for the
  same consistent **`M_uu`** as [`ContinuumKernel`](@ref) / [`BiotPoroelasticKernel`](@ref);
  default `0` skips.

Weak form (steady stiffness; mechanical and diagonal thermal / Darcy blocks
as before):

    ∫ ℂ:ε(u):ε(v) dΩ  − β ∫ T · div(v) dΩ  − α ∫ p · div(v) dΩ  = rhs_u(v)
    − β ∫ u · div(θ) dΩ  + ∫ k_T ∇T · ∇θ dΩ  + κ_tp ∫ ∇p · ∇θ dΩ   = rhs_T(θ)
    − α ∫ q · div(u) dΩ  + ∫ k_p ∇p · ∇q dΩ  + κ_tp ∫ ∇T · ∇q dΩ   = rhs_p(q)

with `β` and `α` as in [`ThermoElasticKernel`](@ref) / [`BiotPoroelasticKernel`](@ref).
Tensors `k_T` and `k_p` come from [`HeatConductivity`](@ref) and
[`HydraulicConductivity`](@ref).

# DOF template

```julia
S = @DOFSet{
    u::DOF{Displacement{3}, Vertex},
    T::DOF{Temperature, Vertex},
    p::DOF{PorePressure, Vertex},
}
kernel = ThermoPoroelasticKernel(
    ContinuumFormulation{ThreeDimensional}(),
    LinearElastic(E = 3.0e10, ν = 0.25),
    HeatConductivity(k = 2.0),
    HydraulicConductivity(K = 1.0e-12),
    2.7e6,  # β (Pa/K): thermo-elastic stress–temperature modulus; often ~1e6–1e8
    0.8,    # α (Biot coefficient, dimensionless)
    0.0,    # storage_S
    0.0,    # kappa_tp
    0.0,    # zeta_tp
    0.0,    # heat_capacity  (ρ c_p on T; optional M_TT)
    0.0,    # density (solid ρ on u; optional M_uu)
)
```
=#

using Tensors

using ..JuliaFEM: AbstractKernel, AbstractFormulation
using ..JuliaFEM: ContinuumFormulation, AbstractContinuumTheory
using ..JuliaFEM: AbstractMaterial, LinearElastic, HeatConductivity, HydraulicConductivity
using ..JuliaFEM: conductivity_tensor, hydraulic_conductivity_tensor
using ..JuliaFEM: AssemblyMaterialWorkspace
using ..JuliaFEM: compute_stress, compute_stiffness_value
import ..JuliaFEM: qpoint_buffer_eltype, update_qpoint_buffer!, evaluate_entry,
                   evaluate_mass_entry,
                   reference_fields, get_field, dofs_per_node
using ..JuliaFEM: DOFLayoutEntry, field_idx, entity_local, component


"""
    ThermoPoroelasticQPBuffer

Per-IP buffer: elasticity tangent `C`, thermal conductivity `k_T`, hydraulic
conductivity `k_p`. Bitstype for the DOF-based / KA paths.
"""
struct ThermoPoroelasticQPBuffer
    C::SymmetricTensor{4,3,Float64,36}
    k_T::SymmetricTensor{2,3,Float64,6}
    k_p::SymmetricTensor{2,3,Float64,6}
end


"""
    ThermoPoroelasticKernel{Theory, MatM, MatT, MatF}

Three-field kernel: `u` (3 DOFs / node) + `T` (1) + `p` (1) = 5 DOFs / node.

# Fields
- `formulation::ContinuumFormulation{Theory}`
- `mech_material::MatM` — solid (`LinearElastic`, …)
- `therm_material::MatT` — [`HeatConductivity`](@ref) (`k_T`)
- `flow_material::MatF` — [`HydraulicConductivity`](@ref) (`k_p`)
- `β::Float64` — thermo-mechanical coupling stress–temperature modulus (`0` ⇒
  decoupled `u`–`T` blocks), same units as `ThermoElasticKernel` (`β`, Pa/K)
- `α::Float64` — Biot coefficient (`0` ⇒ decoupled `u`–`p` blocks)
- `storage_S::Float64` — fluid storage on `p` for `assemble_M!` (`0` skips)
- `kappa_tp::Float64` — symmetric `∇T`–`∇p` stiffness coupling (`≥ 0`)
- `zeta_tp::Float64` — symmetric `T`–`p` L² mass for transient coupling (`≥ 0`)
- `heat_capacity::Float64` — volumetric ``ρ c_p`` on `T` for `M_TT` (`≥ 0`; `0` skips)
- `density::Float64` — solid mass density on `u` for `M_uu` (`≥ 0`; `0` skips)

In `evaluate_mass_entry`, non-zero **`α`** adds **`M_pu`** and **`M_up`**
(negatives of **`K_pu`** and **`K_up`**); non-zero **`β`** adds **`M_Tu`** and
**`M_uT`** (negatives of **`K_Tu`** and **`K_uT`**); non-zero **`density`** adds
**`M_uu`** like [`ContinuumKernel`](@ref).
"""
struct ThermoPoroelasticKernel{Theory<:AbstractContinuumTheory,
                              MatM<:AbstractMaterial,
                              MatT<:HeatConductivity,
                              MatF<:HydraulicConductivity} <: AbstractKernel
    formulation::ContinuumFormulation{Theory}
    mech_material::MatM
    therm_material::MatT
    flow_material::MatF
    β::Float64
    α::Float64
    storage_S::Float64
    kappa_tp::Float64
    zeta_tp::Float64
    heat_capacity::Float64
    density::Float64

    function ThermoPoroelasticKernel(
        formulation::ContinuumFormulation{Theory},
        mech_material::MatM,
        therm_material::MatT,
        flow_material::MatF,
        β::Float64,
        α::Float64,
        storage_S::Float64,
        kappa_tp::Float64,
        zeta_tp::Float64,
        heat_capacity::Float64,
        density::Float64,
    ) where {Theory<:AbstractContinuumTheory,
              MatM<:AbstractMaterial,
              MatT<:HeatConductivity,
              MatF<:HydraulicConductivity}
        α ≥ 0.0 || throw(ArgumentError("Biot coefficient α must be ≥ 0, got α = $α"))
        storage_S ≥ 0.0 ||
            throw(ArgumentError("storage_S must be ≥ 0, got storage_S = $storage_S"))
        kappa_tp ≥ 0.0 ||
            throw(ArgumentError("kappa_tp must be ≥ 0, got kappa_tp = $kappa_tp"))
        zeta_tp ≥ 0.0 ||
            throw(ArgumentError("zeta_tp must be ≥ 0, got zeta_tp = $zeta_tp"))
        heat_capacity ≥ 0.0 || throw(
            ArgumentError("heat_capacity must be ≥ 0, got heat_capacity = $heat_capacity"),
        )
        density ≥ 0.0 ||
            throw(ArgumentError("density must be ≥ 0, got density = $density"))
        return new{Theory, MatM, MatT, MatF}(
            formulation,
            mech_material,
            therm_material,
            flow_material,
            β,
            α,
            storage_S,
            kappa_tp,
            zeta_tp,
            heat_capacity,
            density,
        )
    end
end

"""Ten-argument form sets `density = 0` (omit solid `M_uu`)."""
function ThermoPoroelasticKernel(
    formulation::ContinuumFormulation{Theory},
    mech_material::MatM,
    therm_material::MatT,
    flow_material::MatF,
    β::Float64,
    α::Float64,
    storage_S::Float64,
    kappa_tp::Float64,
    zeta_tp::Float64,
    heat_capacity::Float64,
) where {Theory, MatM, MatT, MatF}
    return ThermoPoroelasticKernel(
        formulation,
        mech_material,
        therm_material,
        flow_material,
        β,
        α,
        storage_S,
        kappa_tp,
        zeta_tp,
        heat_capacity,
        0.0,
    )
end

"""Nine-argument form sets `heat_capacity = 0` and `density = 0`."""
function ThermoPoroelasticKernel(
    formulation::ContinuumFormulation{Theory},
    mech_material::MatM,
    therm_material::MatT,
    flow_material::MatF,
    β::Float64,
    α::Float64,
    storage_S::Float64,
    kappa_tp::Float64,
    zeta_tp::Float64,
) where {Theory, MatM, MatT, MatF}
    return ThermoPoroelasticKernel(
        formulation,
        mech_material,
        therm_material,
        flow_material,
        β,
        α,
        storage_S,
        kappa_tp,
        zeta_tp,
        0.0,
        0.0,
    )
end

function ThermoPoroelasticKernel(
    formulation::ContinuumFormulation{Theory},
    mech_material::MatM,
    therm_material::MatT,
    flow_material::MatF,
    β::Float64,
    α::Float64,
    storage_S::Float64,
) where {Theory, MatM, MatT, MatF}
    return ThermoPoroelasticKernel(
        formulation,
        mech_material,
        therm_material,
        flow_material,
        β,
        α,
        storage_S,
        0.0,
        0.0,
        0.0,
        0.0,
    )
end

function ThermoPoroelasticKernel(
    formulation::ContinuumFormulation{Theory},
    mech_material::MatM,
    therm_material::MatT,
    flow_material::MatF,
    β::Float64,
    α::Float64,
) where {Theory, MatM, MatT, MatF}
    return ThermoPoroelasticKernel(
        formulation, mech_material, therm_material, flow_material, β, α, 0.0,
    )
end

"""Convenience: `β = 0`, `α = 0`, `storage_S = 0` (block-diagonal smoke setup)."""
function ThermoPoroelasticKernel(
    formulation::ContinuumFormulation{Theory},
    mech_material::MatM,
    therm_material::MatT,
    flow_material::MatF,
) where {Theory, MatM, MatT, MatF}
    return ThermoPoroelasticKernel(
        formulation,
        mech_material,
        therm_material,
        flow_material,
        0.0,
        0.0,
        0.0,
    )
end

@inline operator_is_posdef(::ThermoPoroelasticKernel) = false

@inline dofs_per_node(::ThermoPoroelasticKernel) = 5

function get_field(::K) where {K<:ThermoPoroelasticKernel}
    error("$(K) is multi-field — use `local_dof_layout(E)` and " *
          "`elem.dof_indices` instead of `get_field(kernel)`.")
end

@inline qpoint_buffer_eltype(::ThermoPoroelasticKernel) = ThermoPoroelasticQPBuffer

@inline function reference_fields(kernel::ThermoPoroelasticKernel)
    ε_ref = zero(SymmetricTensor{2,3,Float64,6})
    σ_ref, 𝔻_ref, _ = compute_stress(kernel.mech_material, ε_ref, NamedTuple(), 0.0)
    q_ref = zero(Vec{3,Float64})
    k_T = conductivity_tensor(kernel.therm_material)
    k_p = hydraulic_conductivity_tensor(kernel.flow_material)
    return ((σ = σ_ref, 𝔻 = 𝔻_ref, q = q_ref, k_T = k_T, k_p = k_p), NamedTuple())
end

@inline function update_qpoint_buffer!(
    buffer::AbstractVector{ThermoPoroelasticQPBuffer},
    workspace::AssemblyMaterialWorkspace{FieldType, StateType},
    ::ThermoPoroelasticKernel,
) where {FieldType, StateType}
    fields = getfield(workspace, 1)
    @inbounds for q in eachindex(buffer)
        f = fields[q]
        buffer[q] = ThermoPoroelasticQPBuffer(f.𝔻, f.k_T, f.k_p)
    end
    return nothing
end

"""
    evaluate_entry(::ThermoPoroelasticKernel, ...)

Non-zero blocks: `(1,1)` `K_uu`, `(2,2)` `K_TT`, `(3,3)` `K_pp`,
`(1,2)`/`(2,1)` thermo-elastic, `(1,3)`/`(3,1)` Biot, and if `kappa_tp > 0`
the symmetric gradient pair `(2,3)` / `(3,2)`.
"""
@inline function evaluate_entry(
    kernel::ThermoPoroelasticKernel,
    geometry_cache,
    qp_vec::AbstractVector{ThermoPoroelasticQPBuffer},
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
    K_ij = 0.0

    if fi == 1 && fj == 1
        @inbounds for q in 1:n_ips
            ∇N_i = geometry_cache.∇N_data[q, node_i]
            ∇N_j = geometry_cache.∇N_data[q, node_j]
            detJw = geometry_cache.detJ_w[q]
            C = Tensor{4,3}(qp_vec[q].C)
            K_ij += compute_stiffness_value(∇N_i, ∇N_j, C, comp_i, comp_j) * detJw
        end

    elseif fi == 2 && fj == 2
        @inbounds for q in 1:n_ips
            ∇N_i = geometry_cache.∇N_data[q, node_i]
            ∇N_j = geometry_cache.∇N_data[q, node_j]
            detJw = geometry_cache.detJ_w[q]
            k_q = qp_vec[q].k_T
            K_ij += (∇N_i ⋅ k_q ⋅ ∇N_j) * detJw
        end

    elseif fi == 3 && fj == 3
        @inbounds for q in 1:n_ips
            ∇N_i = geometry_cache.∇N_data[q, node_i]
            ∇N_j = geometry_cache.∇N_data[q, node_j]
            detJw = geometry_cache.detJ_w[q]
            k_q = qp_vec[q].k_p
            K_ij += (∇N_i ⋅ k_q ⋅ ∇N_j) * detJw
        end

    elseif fi == 1 && fj == 2
        β = kernel.β
        @inbounds for q in 1:n_ips
            ∇N_i = geometry_cache.∇N_data[q, node_i]
            N_j = geometry_cache.N_data[q, node_j]
            detJw = geometry_cache.detJ_w[q]
            K_ij += -β * ∇N_i[comp_i] * N_j * detJw
        end

    elseif fi == 2 && fj == 1
        β = kernel.β
        @inbounds for q in 1:n_ips
            N_i = geometry_cache.N_data[q, node_i]
            ∇N_j = geometry_cache.∇N_data[q, node_j]
            detJw = geometry_cache.detJ_w[q]
            K_ij += -β * N_i * ∇N_j[comp_j] * detJw
        end

    elseif fi == 1 && fj == 3
        α = kernel.α
        @inbounds for q in 1:n_ips
            ∇N_i = geometry_cache.∇N_data[q, node_i]
            N_j = geometry_cache.N_data[q, node_j]
            detJw = geometry_cache.detJ_w[q]
            K_ij += -α * ∇N_i[comp_i] * N_j * detJw
        end

    elseif fi == 3 && fj == 1
        α = kernel.α
        @inbounds for q in 1:n_ips
            N_i = geometry_cache.N_data[q, node_i]
            ∇N_j = geometry_cache.∇N_data[q, node_j]
            detJw = geometry_cache.detJ_w[q]
            K_ij += -α * N_i * ∇N_j[comp_j] * detJw
        end

    elseif (fi == 2 && fj == 3) || (fi == 3 && fj == 2)
        κ = kernel.kappa_tp
        if κ != 0.0
            @inbounds for q in 1:n_ips
                ∇N_i = geometry_cache.∇N_data[q, node_i]
                ∇N_j = geometry_cache.∇N_data[q, node_j]
                detJw = geometry_cache.detJ_w[q]
                K_ij += κ * (∇N_i ⋅ ∇N_j) * detJw
            end
        end
    end

    return K_ij
end

"""
    evaluate_mass_entry(::ThermoPoroelasticKernel, ...)

Diagonal / cross L² mass: `M_TT` from `heat_capacity` (same as [`HeatKernel`](@ref)),
`M_pp` from `storage_S`, symmetric `M_Tp`/`M_pT` from `zeta_tp`, optional `M_uu`
from `density` (same as [`ContinuumKernel`](@ref)), and when **`α > 0`** the
**`M_pu`** / **`M_up`** pair (negatives of **`K_pu`** / **`K_up`**). When
**`β > 0`**, the **`M_Tu`** / **`M_uT`** pair are negatives of **`K_Tu`** /
**`K_uT`**. Blocks compose additively.
"""
@inline function evaluate_mass_entry(
    kernel::ThermoPoroelasticKernel,
    geometry_cache,
    qp_buffer,
    layout_i::DOFLayoutEntry,
    layout_j::DOFLayoutEntry,
)
    fi = field_idx(layout_i)
    fj = field_idx(layout_j)
    node_i = entity_local(layout_i)
    node_j = entity_local(layout_j)
    n_ips = length(geometry_cache.detJ_w)
    M_ij = 0.0
    S = kernel.storage_S
    ζ = kernel.zeta_tp
    ρcp = kernel.heat_capacity
    ρs = kernel.density

    if fi == 1 && fj == 1 && ρs != 0.0
        comp_i = component(layout_i)
        comp_j = component(layout_j)
        if comp_i == comp_j
            @inbounds for q in 1:n_ips
                N_i   = geometry_cache.N_data[q, node_i]
                N_j   = geometry_cache.N_data[q, node_j]
                detJw = geometry_cache.detJ_w[q]
                M_ij += ρs * N_i * N_j * detJw
            end
        end
    end

    if fi == 2 && fj == 2 && ρcp != 0.0
        @inbounds for q in 1:n_ips
            N_i = geometry_cache.N_data[q, node_i]
            N_j = geometry_cache.N_data[q, node_j]
            detJw = geometry_cache.detJ_w[q]
            M_ij += ρcp * N_i * N_j * detJw
        end
    end

    if fi == 3 && fj == 3 && S != 0.0
        @inbounds for q in 1:n_ips
            N_i = geometry_cache.N_data[q, node_i]
            N_j = geometry_cache.N_data[q, node_j]
            detJw = geometry_cache.detJ_w[q]
            M_ij += S * N_i * N_j * detJw
        end
    end

    if ((fi == 2 && fj == 3) || (fi == 3 && fj == 2)) && ζ != 0.0
        @inbounds for q in 1:n_ips
            N_i = geometry_cache.N_data[q, node_i]
            N_j = geometry_cache.N_data[q, node_j]
            detJw = geometry_cache.detJ_w[q]
            M_ij += ζ * N_i * N_j * detJw
        end
    end

    if fi == 3 && fj == 1
        α = kernel.α
        if α != 0.0
            comp_j = component(layout_j)
            @inbounds for q in 1:n_ips
                N_i   = geometry_cache.N_data[q, node_i]
                ∇N_j  = geometry_cache.∇N_data[q, node_j]
                detJw = geometry_cache.detJ_w[q]
                M_ij += α * N_i * ∇N_j[comp_j] * detJw
            end
        end
    end

    if fi == 2 && fj == 1
        β = kernel.β
        if β != 0.0
            comp_j = component(layout_j)
            @inbounds for q in 1:n_ips
                N_i   = geometry_cache.N_data[q, node_i]
                ∇N_j  = geometry_cache.∇N_data[q, node_j]
                detJw = geometry_cache.detJ_w[q]
                M_ij += β * N_i * ∇N_j[comp_j] * detJw
            end
        end
    end

    if fi == 1 && fj == 3
        α = kernel.α
        if α != 0.0
            comp_i = component(layout_i)
            @inbounds for q in 1:n_ips
                ∇N_i  = geometry_cache.∇N_data[q, node_i]
                N_j   = geometry_cache.N_data[q, node_j]
                detJw = geometry_cache.detJ_w[q]
                M_ij += α * ∇N_i[comp_i] * N_j * detJw
            end
        end
    end

    if fi == 1 && fj == 2
        β = kernel.β
        if β != 0.0
            comp_i = component(layout_i)
            @inbounds for q in 1:n_ips
                ∇N_i  = geometry_cache.∇N_data[q, node_i]
                N_j   = geometry_cache.N_data[q, node_j]
                detJw = geometry_cache.detJ_w[q]
                M_ij += β * ∇N_i[comp_i] * N_j * detJw
            end
        end
    end

    return M_ij
end
