# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

#=
Heat-conduction kernel — the second concrete kernel in the codebase.

Its sole purpose right now is to prove the microkernel contract is
genuinely kernel-agnostic: the same `DOFBasedCOOCache` constructor
(via the new `reference_fields(kernel)` hook) and the same `assemble!`
/ `apply_K!` paths must accept this kernel without continuum-specific
detours.

Weak form (steady-state, isotropic linear conduction):

    a(T, v) = ∫ ∇v · k · ∇T  dV    →    K_ij = ∫ ∇N_i · k · ∇N_j  dV

with `k` symmetric positive-definite (we cache it as a
`SymmetricTensor{2,3,Float64,6}`; the isotropic case is just
`k = κ · I`).

The same assembly implements **steady primal Darcy flow potential**
`-∇·(K∇p)=f` when paired with [`HydraulicConductivity`](@ref),
[`PressurePotential`](@ref), and [`DarcyPotentialKernel`](@ref); see
`src/domains/darcy/potential.jl`.

Differences from `ContinuumKernel`:

* one DOF per node (`Temperature` or `PressurePotential`) → `dofs_per_node == 1`
* per-IP buffer is the (small) 2nd-order conductivity tensor instead
  of the 4th-order elasticity tensor
* `evaluate_entry` returns `∇N_i · k · ∇N_j * detJw` summed over IPs,
  with no `(α, β)` component decomposition — `component(layout)` is
  always `1` and is ignored
=#

using Tensors

using ..JuliaFEM: AbstractKernel, AbstractFormulation, AbstractMaterial, AbstractField
using ..JuliaFEM: ContinuumFormulation, FullThreeD
using ..JuliaFEM: Temperature, MoistureContent, PressurePotential, dofs_per_node, get_field
using ..JuliaFEM: HeatConductivity, MoistureDiffusivity, HydraulicConductivity,
    ElementWiseScalarDiffusion, scalar_diffusion_tensor
using ..JuliaFEM: AssemblyMaterialWorkspace
import ..JuliaFEM: qpoint_buffer_eltype, update_qpoint_buffer!, evaluate_entry,
                   evaluate_mass_entry,
                   reference_fields, get_field, dofs_per_node
using ..JuliaFEM: DOFLayoutEntry, entity_local, component


"""
    HeatKernel{Theory, Mat, Fld} <: AbstractKernel

Domain kernel for steady-state **scalar diffusion**: linear heat conduction
(`HeatConductivity` + [`Temperature`](@ref)), moisture diffusion
([`MoistureDiffusivity`](@ref) + [`MoistureContent`](@ref)), primal Darcy potential
([`HydraulicConductivity`](@ref) + [`PressurePotential`](@ref)), or
[`ElementWiseScalarDiffusion`](@ref) for piecewise-constant ``k`` or ``K`` per
volume element.

Reuses `ContinuumFormulation{Theory}` (the geometric formulation is
field-agnostic; `FullThreeD` works equally well for displacement,
temperature, potential, …).

# Fields
- `formulation::ContinuumFormulation{Theory}` — geometric formulation
- `material::Mat` — [`HeatConductivity`](@ref), [`MoistureDiffusivity`](@ref),
  [`HydraulicConductivity`](@ref), or [`ElementWiseScalarDiffusion`](@ref)
- `field::Fld` — matching scalar vertex field (`Temperature`, `MoistureContent`, or `PressurePotential`)

# Example

```julia
kernel = HeatKernel(
    ContinuumFormulation{FullThreeD}(),
    HeatConductivity(k = 401.0),
)
```
"""
struct HeatKernel{Theory, Mat <: AbstractMaterial, Fld <: AbstractField} <: AbstractKernel
    formulation::ContinuumFormulation{Theory}
    material::Mat
    field::Fld
    # Volumetric heat capacity ρ·c_p [J / (m³·K)]; powers the
    # *capacity* matrix `C = ∫ ρ c_p N_i N_j dV` accessed via the same
    # `evaluate_mass_entry` / `apply_M!` / `assemble_M!` machinery used
    # by `ContinuumKernel.density`. Default `0.0` keeps the
    # steady-state path unchanged.
    heat_capacity::Float64
end

function HeatKernel(
    formulation::ContinuumFormulation{Theory},
    material::HeatConductivity,
    field::Temperature = Temperature();
    heat_capacity::Float64 = 0.0,
) where {Theory}
    return HeatKernel{Theory, HeatConductivity, Temperature}(formulation, material, field, heat_capacity)
end

function HeatKernel(
    formulation::ContinuumFormulation{Theory},
    material::MoistureDiffusivity,
    field::MoistureContent = MoistureContent();
    heat_capacity::Float64 = 0.0,
) where {Theory}
    heat_capacity != 0.0 && throw(
        ArgumentError("Moisture diffusion is steady; heat_capacity must be 0 (got $heat_capacity)"),
    )
    return HeatKernel{Theory, MoistureDiffusivity, MoistureContent}(
        formulation, material, field, 0.0,
    )
end

function HeatKernel(
    formulation::ContinuumFormulation{Theory},
    material::HydraulicConductivity,
    field::PressurePotential = PressurePotential();
    heat_capacity::Float64 = 0.0,
) where {Theory}
    heat_capacity != 0.0 && throw(
        ArgumentError(
            "Hydraulic potential kernel is steady primal Darcy; heat_capacity must stay 0 (got $heat_capacity)",
        ),
    )
    return HeatKernel{Theory, HydraulicConductivity, PressurePotential}(
        formulation,
        material,
        field,
        heat_capacity,
    )
end

function HeatKernel(
    ::ContinuumFormulation{Theory},
    ::HeatConductivity,
    ::PressurePotential;
    heat_capacity::Float64 = 0.0,
) where {Theory}
    throw(
        ArgumentError(
            "HeatKernel: HeatConductivity requires Temperature field; use HydraulicConductivity with PressurePotential for Darcy potential",
        ),
    )
end

function HeatKernel(
    ::ContinuumFormulation{Theory},
    ::HydraulicConductivity,
    ::Temperature;
    heat_capacity::Float64 = 0.0,
) where {Theory}
    throw(
        ArgumentError(
            "HeatKernel: HydraulicConductivity requires PressurePotential field (see DarcyPotentialKernel)",
        ),
    )
end

function HeatKernel(
    ::ContinuumFormulation{Theory},
    ::MoistureDiffusivity,
    ::Temperature;
    heat_capacity::Float64 = 0.0,
) where {Theory}
    throw(
        ArgumentError(
            "HeatKernel: MoistureDiffusivity requires MoistureContent field",
        ),
    )
end

function HeatKernel(
    ::ContinuumFormulation{Theory},
    ::HeatConductivity,
    ::MoistureContent;
    heat_capacity::Float64 = 0.0,
) where {Theory}
    throw(
        ArgumentError(
            "HeatKernel: HeatConductivity requires Temperature field",
        ),
    )
end

function HeatKernel(
    ::ContinuumFormulation{Theory},
    ::HydraulicConductivity,
    ::MoistureContent;
    heat_capacity::Float64 = 0.0,
) where {Theory}
    throw(
        ArgumentError(
            "HeatKernel: HydraulicConductivity requires PressurePotential field",
        ),
    )
end

function HeatKernel(
    formulation::ContinuumFormulation{Theory},
    material::ElementWiseScalarDiffusion,
    field::Temperature = Temperature();
    heat_capacity::Float64 = 0.0,
) where {Theory}
    return HeatKernel{Theory, ElementWiseScalarDiffusion, Temperature}(
        formulation,
        material,
        field,
        heat_capacity,
    )
end

function HeatKernel(
    formulation::ContinuumFormulation{Theory},
    material::ElementWiseScalarDiffusion,
    field::PressurePotential;
    heat_capacity::Float64 = 0.0,
) where {Theory}
    heat_capacity != 0.0 && throw(
        ArgumentError(
            "HeatKernel: ElementWiseScalarDiffusion with PressurePotential is steady Darcy; heat_capacity must be 0 (got $heat_capacity)",
        ),
    )
    return HeatKernel{Theory, ElementWiseScalarDiffusion, PressurePotential}(
        formulation,
        material,
        field,
        heat_capacity,
    )
end

# ----------------------------------------------------------------------------
# Field interface
# ----------------------------------------------------------------------------

@inline get_field(kernel::HeatKernel)     = kernel.field
@inline dofs_per_node(::HeatKernel)       = 1

# ----------------------------------------------------------------------------
# Microkernel contract — kernel-agnostic surface for the DOF-based assembler
# ----------------------------------------------------------------------------

@inline qpoint_buffer_eltype(::HeatKernel) = SymmetricTensor{2,3,Float64,6}

"""
    reference_fields(kernel::HeatKernel)

Per-IP reference values for steady-state linear heat conduction:

* `q::Vec{3,Float64}`             — heat flux at zero gradient is zero
* `k::SymmetricTensor{2,3,Float64,6}` — conductivity tensor, the only
  thing the microkernel actually reads

Names match `required_material_fields(::Thermal{3})` so the same
`(q, k)` workspace shape is reused for heat, Darcy potential, and moisture diffusion.
"""
@inline function reference_fields(
    kernel::HeatKernel{Theory, Mat, Fld},
) where {Theory, Mat <: Union{HeatConductivity, HydraulicConductivity, MoistureDiffusivity}, Fld}
    q_ref = zero(Vec{3,Float64})
    k_ref = scalar_diffusion_tensor(kernel.material)
    return ((q = q_ref, k = k_ref), NamedTuple())
end

@inline function reference_fields(
    kernel::HeatKernel{Theory, ElementWiseScalarDiffusion, Fld},
) where {Theory, Fld}
    q_ref = zero(Vec{3,Float64})
    v = kernel.material.λ_by_elem
    isempty(v) && error("ElementWiseScalarDiffusion: λ_by_elem must be non-empty")
    k_ref = v[1] * one(SymmetricTensor{2,3,Float64,6})
    return ((q = q_ref, k = k_ref), NamedTuple())
end

"""
    update_qpoint_buffer!(buffer, workspace, ::HeatKernel)

Pull the conductivity tensor `k` for every IP out of the per-element
material workspace into the per-IP buffer the assembler keeps in its
`qp_buffers` matrix. Allocation-free.
"""
@inline function update_qpoint_buffer!(
    buffer::AbstractVector{SymmetricTensor{2,3,Float64,6}},
    workspace::AssemblyMaterialWorkspace{FieldType, StateType},
    kernel::HeatKernel{Theory, Mat, Fld},
) where {
    Theory,
    Mat <: Union{HeatConductivity, HydraulicConductivity, MoistureDiffusivity},
    Fld,
    FieldType,
    StateType,
}
    fields = getfield(workspace, 1)
    n      = length(buffer)
    @inbounds for q in 1:n
        buffer[q] = fields[q].k
    end
    return nothing
end

"""
    update_qpoint_buffer!(buffer, workspace, kernel::HeatKernel{*,ElementWiseScalarDiffusion,*}, eid)

Element-wise isotropic conductivity `λ_by_elem[eid] · I` at every IP.
The DOF-based assembler Pass 1 calls this 4-arg method only for this
material (see `_dof_based_fill_qpoint_buffer!` in `dof_based_coo.jl`);
other kernels use the 3-arg `update_qpoint_buffer!` only.
"""
@inline function update_qpoint_buffer!(
    buffer::AbstractVector{SymmetricTensor{2,3,Float64,6}},
    workspace::AssemblyMaterialWorkspace{FieldType, StateType},
    kernel::HeatKernel{Theory, ElementWiseScalarDiffusion, Fld},
    eid::Int,
) where {Theory, Fld, FieldType, StateType}
    λvec = kernel.material.λ_by_elem
    Kscalar = @inbounds λvec[eid]
    kt = Kscalar * one(SymmetricTensor{2,3,Float64,6})
    n = length(buffer)
    @inbounds for q in 1:n
        buffer[q] = kt
    end
    return nothing
end

"""
    evaluate_entry(kernel::HeatKernel, geometry_cache,
                   k_vec::AbstractVector{SymmetricTensor{2,3,Float64,6}},
                   layout_i::DOFLayoutEntry, layout_j::DOFLayoutEntry,
                   elem_id::Int) -> Float64

Heat-conduction microkernel for the DOF-based assembler.

Single-field scalar vertex unknown (`Temperature`, `MoistureContent`, or `PressurePotential`) so `component(layout)` is
always `1` and is ignored. Sums `∇N_i · k · ∇N_j * detJ·w` over
quadrature points and returns the scalar `K[i, j]`. By construction
SPD when `k` is SPD. The volume kernel ignores `elem_id`.

Allocation-free; same access pattern as `ContinuumKernel.evaluate_entry`
so it inherits the SoA-friendly batched geometry layout.
"""
@inline function evaluate_entry(
    kernel::HeatKernel,
    geometry_cache,
    k_vec::AbstractVector{<:SymmetricTensor{2,3}},
    layout_i::DOFLayoutEntry,
    layout_j::DOFLayoutEntry,
    ::Int,
)
    node_i = entity_local(layout_i)
    node_j = entity_local(layout_j)

    F = eltype(geometry_cache.detJ_w)
    K_ij = zero(F)
    n_ips = length(geometry_cache.detJ_w)
    @inbounds for q in 1:n_ips
        ∇N_i  = geometry_cache.∇N_data[q, node_i]
        ∇N_j  = geometry_cache.∇N_data[q, node_j]
        detJw = geometry_cache.detJ_w[q]
        k_q   = k_vec[q]
        K_ij += (∇N_i ⋅ k_q ⋅ ∇N_j) * detJw
    end
    return K_ij
end

"""
    evaluate_mass_entry(kernel::HeatKernel, geometry_cache, qp_buffer,
                        layout_i, layout_j) -> Float64

Heat capacity (consistent) matrix microkernel:

    C[i, j] = (ρ·c_p) · Σ_q N_i(q) · N_j(q) · detJ·w(q)

Returns `0.0` when `kernel.heat_capacity == 0`. Same SoA access pattern
as `evaluate_entry` so the matrix-free `apply_M!` and the assembled
`assemble_M!` benefit from the same N_data/detJ_w batches.
"""
@inline function evaluate_mass_entry(
    kernel::HeatKernel,
    geometry_cache,
    qp_buffer,
    layout_i::DOFLayoutEntry,
    layout_j::DOFLayoutEntry,
)
    F   = eltype(geometry_cache.detJ_w)
    ρcp = F(kernel.heat_capacity)
    if ρcp == zero(F)
        return zero(F)
    end

    node_i = entity_local(layout_i)
    node_j = entity_local(layout_j)

    M_ij  = zero(F)
    n_ips = length(geometry_cache.detJ_w)
    @inbounds for q in 1:n_ips
        N_i   = geometry_cache.N_data[q, node_i]
        N_j   = geometry_cache.N_data[q, node_j]
        detJw = geometry_cache.detJ_w[q]
        M_ij += N_i * N_j * detJw
    end
    return ρcp * M_ij
end
