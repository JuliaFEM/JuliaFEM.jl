# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

#=
Three-field Hu–Washizu mixed formulation for small-strain linear elasticity
(3D): independent displacement `u` (vertex), strain `ε̃` (cell, symmetric
Voigt 6-pack), and stress `σ̃` (cell, symmetric Voigt 6-pack).

DOF layout (element template `S`); field order is fixed:

    `@DOFSet{
        u::DOF{Displacement{3}, Vertex},
        eps::DOF{SymmetricTensor{2,3}, Cell},
        sig::DOF{SymmetricTensor{2,3}, Cell},
    }`

with `eps` ≡ independent strain `ε̃` (field index 2) and `sig` ≡ `σ̃`
(field index 3).

Stationary linear weak equations (no body force in `K`; same `B` / Voigt
units as [`HellingerReissnerKernel`](@ref)):

1. ε̃-equation: ∫ (C:ε̃ − σ̃) : δε̃ dΩ = 0
2. σ̃-equation: ∫ (ε̃ − ε(u)) : δσ̃ dΩ = 0
3. u-equation: ∫ σ̃ : ε(v) dΩ (load on RHS in full models)

Discrete blocks (piecewise constant `ε̃`, `σ̃` on the cell; `E_k` Voigt
unit tensors; `M_{ab} = E_a : C : E_b`, `G_{ab} = E_a : E_b`, `vol` =
element volume):

| block | value |
| ----- | ----- |
| `K_uu`, `K_uε`, `K_εu` | `0` |
| `K_uσ` | `+Σ_q E_b : B_{i,α} detJ·w` |
| `K_σu` | `−Σ_q E_a : B_{j,β} detJ·w` |
| `K_εε` | `+ vol · M` |
| `K_εσ` | `− vol · G` |
| `K_σε` | `+ vol · G` |
| `K_σσ` | `0` |

With this Galerkin grouping, `K_{uσ} + K_{σu}' = 0` and the ε̃–σ̃ blocks
match in transpose pairs; the assembled `K` is symmetric indefinite.
Only [`LinearElastic`](@ref) is supported; `M` and `G` are built once at
construction from `C` at zero strain.
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

const _HW_VOIGT_UNITS = ntuple(
    c -> SymmetricTensor{2,3}(ntuple(i -> Float64(i == c), 6)),
    6,
)::NTuple{6,SymmetricTensor{2,3,Float64,6}}

@inline _hw_unit_symmetric_second(c::Int) = @inbounds _HW_VOIGT_UNITS[c]

function _hw_voigt_M_G(C::SymmetricTensor{4,3,Float64,36})
    Es = _HW_VOIGT_UNITS
    Mm = MMatrix{6,6,Float64}(undef)
    Gm = MMatrix{6,6,Float64}(undef)
    @inbounds for j in 1:6, i in 1:6
        Mm[i, j] = dcontract(Es[i], dcontract(C, Es[j]))
        Gm[i, j] = dcontract(Es[i], Es[j])
    end
    return SMatrix(Mm), SMatrix(Gm)
end

@inline function _hw_B_symmetric(∇N::Vec{3,Float64}, comp_u::Int)
    eα = basevec(Vec{3,Float64}, comp_u)
    h = 0.5
    B = h * (∇N ⊗ eα + eα ⊗ ∇N)
    return symmetric(B)
end

"""
    HuWashizuKernel{Theory}

Three-field Hu–Washizu kernel: `u` (vertex) + `ε̃` (cell) + `σ̃` (cell), each
`SymmetricTensor{2,3}` field using the same Voigt layout as
[`HellingerReissnerKernel`](@ref). See the file-level docstring for weak
equations and `K` blocks.

Only [`LinearElastic`](@ref) is supported.

# Example

```julia
S = @DOFSet{
    u::DOF{Displacement{3}, Vertex},
    eps::DOF{SymmetricTensor{2,3}, Cell},
    sig::DOF{SymmetricTensor{2,3}, Cell},
}
kernel = HuWashizuKernel(
    ContinuumFormulation{FullThreeD}(),
    LinearElastic(E = 210e9, ν = 0.3),
)
```
"""
struct HuWashizuKernel{Theory<:AbstractContinuumTheory} <: AbstractKernel
    formulation::ContinuumFormulation{Theory}
    material::LinearElastic
    """`M_{ab} = E_a : ℂ : E_b` (per unit volume; multiply by `vol` in `K_εε`)."""
    M::SMatrix{6,6,Float64,36}
    """`G_{ab} = E_a : E_b` (identity in Voigt metric)."""
    G::SMatrix{6,6,Float64,36}
end

function HuWashizuKernel(
    formulation::ContinuumFormulation{Theory},
    material::LinearElastic,
) where {Theory<:AbstractContinuumTheory}
    ε_ref = zero(SymmetricTensor{2,3,Float64,6})
    _, C, _ = compute_stress(material, ε_ref, NamedTuple(), 0.0)
    M, G = _hw_voigt_M_G(C)
    return HuWashizuKernel{Theory}(formulation, material, M, G)
end

# Three-field u–ε–σ saddle-point system; matrix-free K is symmetric indefinite.
@inline operator_is_posdef(::HuWashizuKernel) = false

function get_field(::K) where {K<:HuWashizuKernel}
    error("$(K) is three-field Hu–Washizu — use `local_dof_layout(E)` and " *
          "`elem.dof_indices`; do not call `get_field(kernel)`.")
end

# Upper bound for `(ndofs_per_elem / nnodes)` on linear Tet4:
# `(12 + 6 + 6) / 4 = 6`.
@inline dofs_per_node(::HuWashizuKernel) = 6

@inline qpoint_buffer_eltype(::HuWashizuKernel) = SymmetricTensor{4,3,Float64,36}

@inline function reference_fields(kernel::HuWashizuKernel)
    ε_ref = zero(SymmetricTensor{2,3,Float64,6})
    σ_ref, 𝔻_ref, _ = compute_stress(kernel.material, ε_ref, NamedTuple(), 0.0)
    return ((σ = σ_ref, 𝔻 = 𝔻_ref), NamedTuple())
end

@inline function update_qpoint_buffer!(
    buffer::AbstractVector{SymmetricTensor{4,3,Float64,36}},
    workspace::AssemblyMaterialWorkspace{FieldType, StateType},
    ::HuWashizuKernel,
) where {FieldType, StateType}
    fields = getfield(workspace, 1)
    extract_tangent!(buffer, fields, FieldType)
    return nothing
end

"""
    evaluate_entry(kernel::HuWashizuKernel, geometry_cache, 𝔻_vec, layout_i, layout_j, elem_id::Int)

Volume kernel; `elem_id` is unused. See the file-level docstring for the
`(field_i, field_j)` block table (field indices `1 = u`, `2 = ε̃`, `3 = σ̃`).
"""
@inline function evaluate_entry(
    kernel::HuWashizuKernel,
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
    vol = 0.0
    @inbounds for q in 1:n_ips
        vol += geometry_cache.detJ_w[q]
    end

    if fi == 1 && fj == 1
        return 0.0
    elseif fi == 1 && fj == 2
        return 0.0
    elseif fi == 1 && fj == 3
        Eσ = _hw_unit_symmetric_second(comp_j)
        K_ij = 0.0
        @inbounds for q in 1:n_ips
            ∇N_i = geometry_cache.∇N_data[q, node_i]
            detJw = geometry_cache.detJ_w[q]
            B = _hw_B_symmetric(∇N_i, comp_i)
            K_ij += dcontract(Eσ, B) * detJw
        end
        return K_ij

    elseif fi == 2 && fj == 1
        return 0.0
    elseif fi == 2 && fj == 2
        return vol * kernel.M[comp_i, comp_j]
    elseif fi == 2 && fj == 3
        return -vol * kernel.G[comp_i, comp_j]

    elseif fi == 3 && fj == 1
        Eσ = _hw_unit_symmetric_second(comp_i)
        K_ij = 0.0
        @inbounds for q in 1:n_ips
            ∇N_j = geometry_cache.∇N_data[q, node_j]
            detJw = geometry_cache.detJ_w[q]
            B = _hw_B_symmetric(∇N_j, comp_j)
            K_ij -= dcontract(Eσ, B) * detJw
        end
        return K_ij

    elseif fi == 3 && fj == 2
        return vol * kernel.G[comp_i, comp_j]
    else  # fi == 3 && fj == 3
        return 0.0
    end
end

@inline evaluate_mass_entry(
    ::HuWashizuKernel,
    geometry_cache,
    qp_buffer,
    layout_i::DOFLayoutEntry,
    layout_j::DOFLayoutEntry,
) = 0.0
