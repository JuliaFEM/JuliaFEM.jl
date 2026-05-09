# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

#=
Mixed lowest-order Darcy on Tet4: RT₀ flux (`DOF{RT0FaceFlux, Face}`) + P₀ pressure
(`DOF{Float64, Cell}`).

Weak form (steady, zero source):

  ∫ K⁻¹ u · v dΩ − ∫ p ∇·v dΩ = 0
  ∫ q ∇·u dΩ        = 0

with **Piola-style RT₀ basis in physical space** on each tet:

  φᵢ(x) = (1/(3V)) (x − xᵢᵒᵖᵖ),

where `V` is element volume and `xᵢᵒᵖᵖ` is the vertex opposite local face `i`
(`faces(::Tet4)` ordering). Then `∇·φᵢ = 1/V` and `∫_K ∇·φᵢ dV = 1`, so the
discrete divergence blocks between face flux and cell pressure are **±1** per
element (pressure gauge required).

Hydraulic resistance in the mass block is **`inv_K`**, the inverse conductivity
tensor (`SymmetricTensor{2,3}`): isotropic `inv_k = 1/K` is `inv_K = inv_k · I`.
The mass block uses `GaussLegendre{2}` on the reference tet (exact for φᵢ·φⱼ,
quadratic). Hex8 uses [`DarcyMixedHex8RT0P0Kernel`](@ref).

Boundary conditions (posture for drivers, not enforced inside the kernel):

  * **Pressure gauge / Dirichlet pressure.** [`DOF{Float64, Cell}`](@ref) `p` is one unknown
    per tet; pin one cell pressure (e.g. [`default_pressure_gauge_dof`](@ref) with
    `field_pressure = 2`) or impose [`PenaltyDirichlet`](@ref) / [`EliminatedDirichlet`](@ref)
    on selected cell `p` DOFs when a boundary head is known.

  * **Prescribed normal flux (Neumann in the primal potential formulation).** The RT₀ face
    scalar is the natural flux unknown; impose its value with [`PenaltyDirichlet`](@ref) or
    [`EliminatedDirichlet`](@ref) on the corresponding global face DOF. Resolve
    `(mesh node triple) → facet_gid` with [`tet_facet_gid_from_corners`](@ref), then the flux
    DOF with [`global_facet_dof(handler, 1, facet_gid)`](@ref) when `σ` is the first field.

  * **[`SurfaceLoad`](@ref)** integrates scalar flux against **nodal** test functions on the
boundary; it does **not** target [`RT0FaceFlux`](@ref) facet DOFs. Use
[`MixedDarcyTet4BoundaryNormalFluxLoad`](@ref) / [`MixedDarcyHex8BoundaryNormalFluxLoad`](@ref)
for a boundary-normal flux density
    against RT₀ test functions on Tet4, or essential flux BCs as above.

Stress-gallery queue (same checklist as mixed Darcy follow-ups): Taylor–Hood,
Nédélec, Hu–Washizu solver story, mortar quadrature on interfaces, hp facet
numbering; wedge / pyramid extensions for mixed RT₀–P₀.
=#

using Tensors

using ..JuliaFEM: AbstractKernel, AssemblyMaterialWorkspace, HydraulicConductivity
using ..JuliaFEM: Tetrahedron, GaussLegendre, get_quadrature_points
using ..JuliaFEM: operator_is_posdef
import ..JuliaFEM: qpoint_buffer_eltype, update_qpoint_buffer!, evaluate_entry,
                   evaluate_mass_entry,
                   reference_fields, get_field, dofs_per_node
using ..JuliaFEM: DOFLayoutEntry, field_idx, entity_local, component

"""Shared supertype for lowest-order mixed RT₀–P₀ Darcy kernels (Tet4, Hex8, …)."""
abstract type AbstractDarcyMixedRT0P0Kernel <: AbstractKernel end

const _TET4_GL2_MIXED_DARCY = get_quadrature_points(Tetrahedron, GaussLegendre{2, Float64}())

"""
    DarcyMixedRT0P0Kernel

Mixed RT₀–P₀ Darcy on [`Mesh{4, Tet4}`](@ref): field 1 = scalar face flux unknown
(`DOF{RT0FaceFlux, Face}`), field 2 = cell pressure (`DOF{Float64, Cell}`).

Use `Element{Tet4, Lagrange{1}, S}`; geometry Pass 1 still fills `∇N` / `detJ·w`
from linear P1 coordinates on the tet.

Conductivity enters through **`inv_K`**, the inverse hydraulic conductivity tensor
(`SymmetricTensor{2,3}`). Isotropic `K` gives `inv_K = (1/K) I`. Pass a symmetric
positive definite tensor directly as `inv_K` for anisotropic resistance.

# Example

```julia
S = @DOFSet{σ::DOF{RT0FaceFlux, Face}, p::DOF{Float64, Cell}}
kernel = DarcyMixedRT0P0Kernel(HydraulicConductivity(K = 1.0))
kernel = DarcyMixedRT0P0Kernel(; inv_k = 0.5)                  # inv_K = 0.5 * I
kernel = DarcyMixedRT0P0Kernel(0.25 * one(SymmetricTensor{2,3,Float64,6}))  # explicit inv_K
```

Pin one pressure DOF (e.g. [`default_pressure_gauge_dof`](@ref) with `field_pressure = 2`)
before solving the indefinite system.

# Boundary conditions

See the module note above: pressure on [`PenaltyDirichlet`](@ref) / gauge on cell `p`;
prescribed boundary flux on [`PenaltyDirichlet`](@ref) / [`EliminatedDirichlet`](@ref) on
[`global_facet_dof`](@ref) for the RT₀ face field. [`SurfaceLoad`](@ref) is nodal (primal),
not mixed RT₀; see [`MixedDarcyTet4BoundaryNormalFluxLoad`](@ref) for Tet4 flux density loads.
"""
struct DarcyMixedRT0P0Kernel <: AbstractDarcyMixedRT0P0Kernel
    inv_K::SymmetricTensor{2,3,Float64,6}

    function DarcyMixedRT0P0Kernel(inv_K::SymmetricTensor{2,3,Float64,6})
        new(inv_K)
    end
end

function DarcyMixedRT0P0Kernel(mat::HydraulicConductivity)
    inv_K = (1.0 / mat.K) * one(SymmetricTensor{2,3,Float64,6})
    return DarcyMixedRT0P0Kernel(inv_K)
end

function DarcyMixedRT0P0Kernel(; inv_k::Float64)
    inv_k ≥ 0.0 || throw(ArgumentError("inv_k must be ≥ 0, got inv_k = $inv_k"))
    return DarcyMixedRT0P0Kernel(inv_k * one(SymmetricTensor{2,3,Float64,6}))
end

@inline operator_is_posdef(::AbstractDarcyMixedRT0P0Kernel) = false

function get_field(::AbstractDarcyMixedRT0P0Kernel)
    error("mixed RT₀–P₀ Darcy — use `local_dof_layout(E)` and `elem.dof_indices`.")
end

@inline dofs_per_node(::AbstractDarcyMixedRT0P0Kernel) = 1

@inline qpoint_buffer_eltype(::AbstractDarcyMixedRT0P0Kernel) = Float64

@inline function reference_fields(::AbstractDarcyMixedRT0P0Kernel)
    return ((aux = 0.0,), NamedTuple())
end

@inline function update_qpoint_buffer!(
    ::AbstractVector{Float64},
    ::AssemblyMaterialWorkspace,
    ::AbstractDarcyMixedRT0P0Kernel,
)
    return nothing
end

# Local face `i` uses vertices `faces(Tet4())[i]`; opposite corner indices match `topology/tetrahedra.jl`.
const _TET4_RT0_FACE_OPP_VERTEX = (4, 3, 1, 2)

@inline function _tet4_detJ_signed(X::AbstractVector{V}) where {V<:Vec{3}}
    @inbounds g1 = X[2] - X[1]
    @inbounds g2 = X[3] - X[1]
    @inbounds g3 = X[4] - X[1]
    return dot(g1 × g2, g3)
end

@inline function _tet4_phys_from_ref(X::AbstractVector{V}, ξ::Vec{3}) where {V<:Vec{3}}
    ξ₁ = ξ[1]
    ξ₂ = ξ[2]
    ξ₃ = ξ[3]
    @inbounds return X[1] + ξ₁ * (X[2] - X[1]) + ξ₂ * (X[3] - X[1]) + ξ₃ * (X[4] - X[1])
end

@inline function _rt0_phi_tet4(X::AbstractVector{V}, Vphys::Float64, iface::Int, x::Vec{3}) where {V<:Vec{3}}
    opp = _TET4_RT0_FACE_OPP_VERTEX[iface]
    scale = 1.0 / (3.0 * Vphys)
    @inbounds xopp = X[opp]
    return scale * (x - xopp)
end

function _mass_uu_entry(
    kernel::DarcyMixedRT0P0Kernel,
    X::AbstractVector{V},
    iface_i::Int,
    iface_j::Int,
) where {V<:Vec{3}}
    detJ = _tet4_detJ_signed(X)
    Vphys = abs(detJ) / 6.0
    Vphys > 0.0 || return 0.0

    acc = 0.0
    @inbounds for q in _TET4_GL2_MIXED_DARCY
        xphys = _tet4_phys_from_ref(X, q.coords)
        φi = _rt0_phi_tet4(X, Vphys, iface_i, xphys)
        φj = _rt0_phi_tet4(X, Vphys, iface_j, xphys)
        acc += (φi ⋅ (kernel.inv_K ⋅ φj)) * abs(detJ) * q.weight
    end
    return acc
end

"""
    evaluate_entry(kernel::DarcyMixedRT0P0Kernel, geometry_cache, qp_buffer, layout_i, layout_j, elem_id)

| `(field_i, field_j)` | block | value |
| -------------------- | ----- | ----- |
| (1, 1)               | Kᵤᵤ   | `∫ φᵢ · inv_K · φⱼ dV` (Gauss–Legendre 4-point on ref. tet) |
| (1, 2)               | Kᵤₚ   | `-∫ ψᵖ ∇·φᵢ = -1` |
| (2, 1)               | Kₚᵤ   | `+∫ ψᵑ ∇·φⱼ = +1` |
| (2, 2)               | Kₚₚ   | `0` |
"""
@inline function evaluate_entry(
    kernel::DarcyMixedRT0P0Kernel,
    geometry_cache,
    ::AbstractVector{Float64},
    layout_i::DOFLayoutEntry,
    layout_j::DOFLayoutEntry,
    ::Int,
)
    fi = field_idx(layout_i)
    fj = field_idx(layout_j)

    X = geometry_cache.X

    if fi == 1 && fj == 1
        iface_i = Int(entity_local(layout_i))
        iface_j = Int(entity_local(layout_j))
        component(layout_i) == component(layout_j) || return 0.0
        return _mass_uu_entry(kernel, X, iface_i, iface_j)

    elseif fi == 1 && fj == 2
        component(layout_i) == component(layout_j) || return 0.0
        return -1.0

    elseif fi == 2 && fj == 1
        component(layout_i) == component(layout_j) || return 0.0
        return 1.0

    else  # fi == 2 && fj == 2
        return 0.0
    end
end

@inline evaluate_mass_entry(
    ::DarcyMixedRT0P0Kernel,
    geometry_cache,
    qp_buffer,
    layout_i::DOFLayoutEntry,
    layout_j::DOFLayoutEntry,
) = 0.0
