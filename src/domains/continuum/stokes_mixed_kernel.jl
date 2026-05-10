# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

#=
Mixed **Stokes** (velocity–pressure) kernel for quasi-steady creeping flow
with Newtonian viscosity `μ` and the same low-order layout as
[`MixedUPKernel`](@ref): vertex vector unknown `u` (stored as
`Displacement{3}` DOFs) and **piecewise-constant** scalar pressure `p` on
`Cell`.

Weak form (steady, no body force in `K` blocks shown here):

    ∫ 2μ ε(u) : ε(v) dΩ − ∫ p div(v) dΩ − ∫ q div(u) dΩ − κ⁻¹ ∫ p q dΩ = rhs

The viscous tangent is **`𝔻 = 2μ 𝕀^sym`** (symmetric fourth-order identity
from `Tensors.jl`), so the `K_uu` block matches [`ContinuumKernel`](@ref)
with that constant operator. The divergence blocks and `K_pp` are
identical to `MixedUPKernel` with the same `inv_bulk = 1/κ` convention
(`0` = incompressible limit; then pin one pressure DOF, e.g. with
[`default_pressure_gauge_dof`](@ref)).

This is **not** Biot poroelasticity (no storage or Darcy diffusion on
`p`); it is the incompressible-flow saddle used as the next mixed model
after solid `u`–`p`.
=#

using Tensors

using ..JuliaFEM: AbstractKernel
using ..JuliaFEM: ContinuumFormulation, AbstractContinuumTheory
using ..JuliaFEM: AssemblyMaterialWorkspace, compute_stiffness_value
import ..JuliaFEM: qpoint_buffer_eltype, update_qpoint_buffer!, evaluate_entry,
                   evaluate_mass_entry,
                   reference_fields, get_field, dofs_per_node
using ..JuliaFEM: DOFLayoutEntry, field_idx, entity_local, component, extract_tangent!


"""
    StokesMixedKernel{Theory}

Newtonian Stokes mixed kernel: `u` (vertex, three components) + scalar
`p` on `Cell`. See the file-level docstring for the weak form.

# Fields
- `formulation::ContinuumFormulation{Theory}` — geometric driver (`ThreeDimensional`, …)
- `μ::Float64` — dynamic viscosity (Stokes: `σ = 2μ ε(u)` with symmetric gradient `ε`)
- `inv_bulk::Float64` — `1/κ` for the `−κ⁻¹ ∫ p q dΩ` term (`0` = incompressible)

# Example

```julia
S = @DOFSet{u::DOF{Displacement{3}, Vertex}, p::DOF{Float64, Cell}}
kernel = StokesMixedKernel(ContinuumFormulation{ThreeDimensional}(); μ = 1.0e-3, inv_bulk = 0.0)
```
"""
struct StokesMixedKernel{Theory<:AbstractContinuumTheory} <: AbstractKernel
    formulation::ContinuumFormulation{Theory}
    μ::Float64
    inv_bulk::Float64
end

function StokesMixedKernel(
    formulation::ContinuumFormulation{Theory};
    μ::Float64,
    inv_bulk::Float64 = 0.0,
) where {Theory<:AbstractContinuumTheory}
    μ > 0.0 || throw(ArgumentError("viscosity μ must be > 0, got μ = $μ"))
    inv_bulk ≥ 0.0 || throw(ArgumentError("inv_bulk must be ≥ 0, got inv_bulk = $inv_bulk"))
    return StokesMixedKernel{Theory}(formulation, μ, inv_bulk)
end

# Saddle-point u–p system; matrix-free K is symmetric indefinite.
@inline operator_is_posdef(::StokesMixedKernel) = false

@inline dofs_per_node(::StokesMixedKernel) = 4

function get_field(::K) where {K<:StokesMixedKernel}
    error("$(K) is mixed Stokes u–p — use `local_dof_layout(E)` and `elem.dof_indices`; " *
          "do not call `get_field(kernel)`.")
end

@inline qpoint_buffer_eltype(::StokesMixedKernel) = SymmetricTensor{4,3,Float64,36}

@inline function reference_fields(kernel::StokesMixedKernel)
    ε_ref = zero(SymmetricTensor{2,3,Float64,6})
    σ_ref = zero(SymmetricTensor{2,3,Float64,6})
    𝕀sym = one(SymmetricTensor{4,3,Float64,36})
    𝔻_ref = (2 * kernel.μ) * 𝕀sym
    return ((σ = σ_ref, 𝔻 = 𝔻_ref), NamedTuple())
end

@inline function update_qpoint_buffer!(
    buffer::AbstractVector{SymmetricTensor{4,3,Float64,36}},
    workspace::AssemblyMaterialWorkspace{FieldType, StateType},
    ::StokesMixedKernel,
) where {FieldType, StateType}
    fields = getfield(workspace, 1)
    extract_tangent!(buffer, fields, FieldType)
    return nothing
end

"""
    evaluate_entry(kernel::StokesMixedKernel, geometry_cache, 𝔻_vec, layout_i, layout_j, elem_id::Int)

Same block table as [`MixedUPKernel`](@ref), with `𝔻 = 2μ 𝕀^sym` from
`reference_fields` / Pass 1 instead of an elastic `LinearElastic` tangent.
`elem_id` is unused.
"""
@inline function evaluate_entry(
    kernel::StokesMixedKernel,
    geometry_cache,
    𝔻_vec::AbstractVector{<:SymmetricTensor{4,3}},
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
            ∇N_i  = geometry_cache.∇N_data[q, node_i]
            ∇N_j  = geometry_cache.∇N_data[q, node_j]
            detJw = geometry_cache.detJ_w[q]
            C = Tensor{4,3}(𝔻_vec[q])
            K_ij += compute_stiffness_value(∇N_i, ∇N_j, C, comp_i, comp_j) * detJw
        end

    elseif fi == 1 && fj == 2
        @inbounds for q in 1:n_ips
            ∇N_i  = geometry_cache.∇N_data[q, node_i]
            detJw = geometry_cache.detJ_w[q]
            K_ij += ∇N_i[comp_i] * detJw
        end

    elseif fi == 2 && fj == 1
        @inbounds for q in 1:n_ips
            ∇N_j  = geometry_cache.∇N_data[q, node_j]
            detJw = geometry_cache.detJ_w[q]
            K_ij += ∇N_j[comp_j] * detJw
        end

    else  # fi == 2 && fj == 2
        ib = kernel.inv_bulk
        if ib == 0.0
            return 0.0
        end
        vol = 0.0
        @inbounds for q in 1:n_ips
            vol += geometry_cache.detJ_w[q]
        end
        K_ij = -ib * vol
    end

    return K_ij
end

@inline evaluate_mass_entry(
    ::StokesMixedKernel,
    geometry_cache,
    qp_buffer,
    layout_i::DOFLayoutEntry,
    layout_j::DOFLayoutEntry,
) = 0.0
