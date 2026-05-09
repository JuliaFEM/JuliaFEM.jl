# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

#=
Classical Hellinger–Reissner mixed formulation for small-strain linear
elasticity (3D): displacement `u` at vertices and a piecewise-constant
symmetric stress tensor `σ` on each cell (`DOF{SymmetricTensor{2,3}, Cell}`).

DOF layout (element template `S`):

    `@DOFSet{u::DOF{Displacement{3}, Vertex}, σ::DOF{SymmetricTensor{2,3}, Cell}}`

Symmetric HR bilinear form (steady, no body-force contribution in `K`):

    a((u,σ), (v,τ)) = −∫ C⁻¹σ : τ dΩ + ∫ σ : ε(v) dΩ + ∫ τ : ε(u) dΩ

with `C` the fourth-order elasticity from `LinearElastic` at each
quadrature point (constant here). The discrete σ–σ block uses the
Voigt-unit tensors implied by `SymmetricTensor{2,3}(ntuple(...))` —
the same ordering as `local_dof_layout` components 1…6 — so the
6×6 matrix `G M⁻¹ G` with `Mᵢⱼ = Eᵢ : C : Eⱼ` and `Gᵢⱼ = Eᵢ : Eⱼ`
matches the double-contraction inner product on `span{E₁…E₆} = Sym(3)`.

The displacement–displacement block vanishes (`K_uu = 0`). The global
system is symmetric indefinite; `MatrixFreeOperator` declares
`isposdef == false` like `MixedUPKernel`.

Currently `material` must be `LinearElastic` (constant `C`); the kernel
precomputes the discrete compliance `G M⁻¹ G` once at construction.
=#

using LinearAlgebra
using StaticArrays: SMatrix, MMatrix
using Tensors
using Tensors: basevec

using ..JuliaFEM: AbstractKernel
using ..JuliaFEM: ContinuumFormulation, AbstractContinuumTheory
using ..JuliaFEM: LinearElastic, AssemblyMaterialWorkspace, compute_stress
import ..JuliaFEM: qpoint_buffer_eltype, update_qpoint_buffer!, evaluate_entry,
                   evaluate_mass_entry,
                   reference_fields, get_field, dofs_per_node
using ..JuliaFEM: DOFLayoutEntry, field_idx, entity_local, component, extract_tangent!

# Unit one-hot symmetric tensors in Tensors.jl Voigt layout (matches `local_dof_layout` components).
const _HR_VOIGT_UNITS = ntuple(
    c -> SymmetricTensor{2,3}(ntuple(i -> Float64(i == c), 6)),
    6,
)::NTuple{6,SymmetricTensor{2,3,Float64,6}}

@inline _hr_unit_stress(c::Int) = @inbounds _HR_VOIGT_UNITS[c]

function _hr_discrete_compliance(C::SymmetricTensor{4,3,Float64,36})
    Es = _HR_VOIGT_UNITS
    Mm = MMatrix{6,6,Float64}(undef)
    Gm = MMatrix{6,6,Float64}(undef)
    @inbounds for j in 1:6, i in 1:6
        Mm[i, j] = dcontract(Es[i], dcontract(C, Es[j]))
        Gm[i, j] = dcontract(Es[i], Es[j])
    end
    Ms = Symmetric(SMatrix(Mm))
    Gs = SMatrix(Gm)
    return Gs * (Ms \ Gs)
end

"""
    HellingerReissnerKernel{Theory}

Hellinger–Reissner mixed kernel: 3D vertex displacement + cell-wise
`SymmetricTensor{2,3}` stress (6 scalars per element). See the file-level
docstring for the weak form.

Only [`LinearElastic`](@ref) is supported; `C` is taken at zero strain and
the discrete `σ`–`σ` block uses `G M⁻¹ G` in the Voigt component basis.

# Example

```julia
S = @DOFSet{u::DOF{Displacement{3}, Vertex}, σ::DOF{SymmetricTensor{2,3}, Cell}}
kernel = HellingerReissnerKernel(
    ContinuumFormulation{FullThreeD}(),
    LinearElastic(E = 210e9, ν = 0.3),
)
```
"""
struct HellingerReissnerKernel{Theory<:AbstractContinuumTheory} <: AbstractKernel
    formulation::ContinuumFormulation{Theory}
    material::LinearElastic
    """Discrete compliance in Voigt σ-components: `K_σσ[i,j] = -vol * σσ[i,j]`."""
    σσ::SMatrix{6,6,Float64,36}
end

function HellingerReissnerKernel(
    formulation::ContinuumFormulation{Theory},
    material::LinearElastic,
) where {Theory<:AbstractContinuumTheory}
    ε_ref = zero(SymmetricTensor{2,3,Float64,6})
    _, C, _ = compute_stress(material, ε_ref, NamedTuple(), 0.0)
    σσ = _hr_discrete_compliance(C)
    return HellingerReissnerKernel{Theory}(formulation, material, σσ)
end

# Mixed u–σ system; matrix-free K is symmetric indefinite.
@inline operator_is_posdef(::HellingerReissnerKernel) = false

function get_field(::K) where {K<:HellingerReissnerKernel}
    error("$(K) is mixed u–σ — use `local_dof_layout(E)` and `elem.dof_indices`; " *
          "do not call `get_field(kernel)`.")
end

# 3 displacement + 6 stress "per cell slot" is not uniform; use an upper
# bound on (total element DOFs / nnodes) for `ElementCache.dofs` sizing.
# `ceil((24+6)/8) = 4` for Hex8; linear Tet4 needs `ceil(18/4) = 5`.
@inline dofs_per_node(::HellingerReissnerKernel) = 5

@inline qpoint_buffer_eltype(::HellingerReissnerKernel) = SymmetricTensor{4,3,Float64,36}

@inline function reference_fields(kernel::HellingerReissnerKernel)
    ε_ref = zero(SymmetricTensor{2,3,Float64,6})
    σ_ref, 𝔻_ref, _ = compute_stress(kernel.material, ε_ref, NamedTuple(), 0.0)
    return ((σ = σ_ref, 𝔻 = 𝔻_ref), NamedTuple())
end

@inline function update_qpoint_buffer!(
    buffer::AbstractVector{SymmetricTensor{4,3,Float64,36}},
    workspace::AssemblyMaterialWorkspace{FieldType, StateType},
    ::HellingerReissnerKernel,
) where {FieldType, StateType}
    fields = getfield(workspace, 1)
    extract_tangent!(buffer, fields, FieldType)
    return nothing
end

@inline function _hr_B_symmetric(∇N::Vec{3,Float64}, comp_u::Int)
    eα = basevec(Vec{3,Float64}, comp_u)
    h = 0.5
    B = h * (∇N ⊗ eα + eα ⊗ ∇N)
    return symmetric(B)
end

"""
    evaluate_entry(kernel::HellingerReissnerKernel, geometry_cache, 𝔻_vec, layout_i, layout_j, elem_id::Int)

Volume kernel; `elem_id` is unused.

| (field_i, field_j) | block | contribution |
| ------------------ | ----- | ------------ |
| (1, 1)             | `K_uu` | `0` |
| (1, 2)             | `K_uσ` | `+ Σ_q E_j : ε(N_i, α_i) detJ_w` |
| (2, 1)             | `K_σu` | `+ Σ_q E_i : ε(N_j, α_j) detJ_w` |
| (2, 2)             | `K_σσ` | `− (Σ_q detJ_w) · (G M⁻¹ G)_{i,j}` |

`E_k` is the unit Voigt basis stress for component `k`; `ε(N,α)` the
symmetric gradient of shape function `N` for displacement component `α`.
"""
@inline function evaluate_entry(
    kernel::HellingerReissnerKernel,
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
        return 0.0

    elseif fi == 1 && fj == 2
        Eσ = _hr_unit_stress(comp_j)
        @inbounds for q in 1:n_ips
            ∇N_i = geometry_cache.∇N_data[q, node_i]
            detJw = geometry_cache.detJ_w[q]
            B = _hr_B_symmetric(∇N_i, comp_i)
            K_ij += dcontract(Eσ, B) * detJw
        end

    elseif fi == 2 && fj == 1
        Eσ = _hr_unit_stress(comp_i)
        @inbounds for q in 1:n_ips
            ∇N_j = geometry_cache.∇N_data[q, node_j]
            detJw = geometry_cache.detJ_w[q]
            B = _hr_B_symmetric(∇N_j, comp_j)
            K_ij += dcontract(Eσ, B) * detJw
        end

    else  # fi == 2 && fj == 2
        vol = 0.0
        @inbounds for q in 1:n_ips
            vol += geometry_cache.detJ_w[q]
        end
        K_ij = -vol * kernel.σσ[comp_i, comp_j]
    end

    return K_ij
end

@inline evaluate_mass_entry(
    ::HellingerReissnerKernel,
    geometry_cache,
    qp_buffer,
    layout_i::DOFLayoutEntry,
    layout_j::DOFLayoutEntry,
) = 0.0
