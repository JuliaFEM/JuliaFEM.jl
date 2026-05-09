# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Mixed lowest-order RT₀–P₀ Darcy on [`Hex8`](@ref): contravariant Piola push-forward of
[`rt0_hex8_reference_basis`](@ref) with trilinear `Hex8` geometry (`get_basis_derivatives`),
same `±1` pressure–flux divergence coupling as Tet4 in [`DarcyMixedRT0P0Kernel`](@ref).

Use `Element{Hex8, Lagrange{1}, S}` with
`S = @DOFSet{σ::DOF{RT0FaceFlux, Face}, p::DOF{Float64, Cell}}`.
"""

using Tensors

using ..JuliaFEM: Hexahedron, Hex8, Lagrange
using ..JuliaFEM: GaussLegendre, get_quadrature_points
using ..JuliaFEM: get_basis_derivatives
using ..JuliaFEM: HydraulicConductivity
using ..JuliaFEM: rt0_hex8_reference_basis, piola_contravariant
import ..JuliaFEM: evaluate_entry, evaluate_mass_entry
using ..JuliaFEM: DOFLayoutEntry, field_idx, entity_local, component

const _HEX_GL2_MIXED_DARCY = get_quadrature_points(Hexahedron, GaussLegendre{2, Float64}())

"""
    DarcyMixedHex8RT0P0Kernel(inv_K::SymmetricTensor{2,3})
    DarcyMixedHex8RT0P0Kernel(mat::HydraulicConductivity)
    DarcyMixedHex8RT0P0Kernel(; inv_k)

Inverse conductivity tensor `inv_K` in the mass block (`∫ φᵢ · inv_K · φⱼ dV`), matching
[`DarcyMixedRT0P0Kernel`](@ref). Quadrature: `GaussLegendre{2}` tensor product on the reference cube.
"""
struct DarcyMixedHex8RT0P0Kernel <: AbstractDarcyMixedRT0P0Kernel
    inv_K::SymmetricTensor{2,3, Float64, 6}

    function DarcyMixedHex8RT0P0Kernel(inv_K::SymmetricTensor{2,3, Float64, 6})
        new(inv_K)
    end
end

function DarcyMixedHex8RT0P0Kernel(mat::HydraulicConductivity)
    inv_K = (1.0 / mat.K) * one(SymmetricTensor{2,3, Float64, 6})
    return DarcyMixedHex8RT0P0Kernel(inv_K)
end

function DarcyMixedHex8RT0P0Kernel(; inv_k::Float64)
    inv_k ≥ 0.0 || throw(ArgumentError("inv_k must be ≥ 0, got inv_k = $inv_k"))
    return DarcyMixedHex8RT0P0Kernel(inv_k * one(SymmetricTensor{2,3, Float64, 6}))
end

function _hex8_mass_uu_entry(
    kernel::DarcyMixedHex8RT0P0Kernel,
    X::AbstractVector{V},
    iface_i::Int,
    iface_j::Int,
) where {V <: Vec{3}}
    length(X) == 8 || return 0.0
    acc = 0.0
    @inbounds for q in _HEX_GL2_MIXED_DARCY
        ξ = q.coords
        dN_dξ = get_basis_derivatives(Hex8(), Lagrange{1}(), ξ)
        J = X[1] ⊗ dN_dξ[1]
        for a in 2:8
            J += X[a] ⊗ dN_dξ[a]
        end
        detJ = det(J)
        detJ == 0.0 && return 0.0
        ψi = rt0_hex8_reference_basis(iface_i, ξ)
        ψj = rt0_hex8_reference_basis(iface_j, ξ)
        φi = piola_contravariant(J, ψi)
        φj = piola_contravariant(J, ψj)
        acc += (φi ⋅ (kernel.inv_K ⋅ φj)) * abs(detJ) * q.weight
    end
    return acc
end

@inline function evaluate_entry(
    kernel::DarcyMixedHex8RT0P0Kernel,
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
        return _hex8_mass_uu_entry(kernel, X, iface_i, iface_j)

    elseif fi == 1 && fj == 2
        component(layout_i) == component(layout_j) || return 0.0
        return -1.0

    elseif fi == 2 && fj == 1
        component(layout_i) == component(layout_j) || return 0.0
        return 1.0

    else
        return 0.0
    end
end

@inline evaluate_mass_entry(
    ::DarcyMixedHex8RT0P0Kernel,
    geometry_cache,
    qp_buffer,
    layout_i::DOFLayoutEntry,
    layout_j::DOFLayoutEntry,
) = 0.0
