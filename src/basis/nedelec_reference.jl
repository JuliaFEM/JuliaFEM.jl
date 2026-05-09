# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Reference lowest-order **Whitney 1-forms** on the **linear reference tetrahedron**
with vertices `(0,0,0)`, `(1,0,0)`, `(0,1,0)`, `(0,0,1)` ([`reference_coordinates(::Tet4)`](@ref)).

Barycentric coordinates `λ₁ … λ₄` sum to `1`; gradients are constant.
Each edge `local_edge ∈ 1:6` carries one Whitney basis

``\\mathbf{N}_{ij} = \\lambda_i \\nabla\\lambda_j - \\lambda_j \\nabla\\lambda_i``,

with `(i,j)` from [`edges(::Tet4)`](@ref).

Push-forward to physical space uses [`piola_covariant`](@ref) with `J = ∂x/∂ξ`.

[`nedelec_hierarchical_tet_edge_reference`](@ref) adds a second **edge-aligned**
reference mode per local edge (product of barycentric factors on that edge with
the Whitney form). This is a convenient hierarchical enrichment toward higher
polynomial edge spaces; it is not a full enumerated `k = 2` Nédélec basis.

This file intentionally stays reference-domain only — assembly kernels can call
these helpers inside quadrature loops once `J` is available.
"""

using Tensors

"""
    nedelec_hierarchical_tet_edge_reference(local_edge::Int, slot::Int, ξ::Vec{3}) -> Vec{3}

Two reference vector fields per `local_edge ∈ 1:6` on the reference `Tet4`:

  - `slot == 1`: [`nedelec_whitney_tet_reference`](@ref) (lowest-order Whitney).
  - `slot == 2`: ``λ_a λ_b \\, \\mathbf{N}^{\\mathrm{Whitney}}`` with `(a,b)` the
    endpoints of that edge in [`edges(::Tet4)`](@ref) order.

`slot` must be `1` or `2`.
"""
@inline function nedelec_hierarchical_tet_edge_reference(local_edge::Int, slot::Int, ξ::Vec{3})
    w = nedelec_whitney_tet_reference(local_edge, ξ)
    slot == 1 && return w
    slot == 2 || throw(ArgumentError("slot must be 1 (Whitney) or 2 (λ_a λ_b enrichment), got $slot"))
    ed = edges(Tet4())[local_edge]
    va, vb = ed.vertices
    ξ₁ = ξ[1]
    ξ₂ = ξ[2]
    ξ₃ = ξ[3]
    λ1 = 1 - ξ₁ - ξ₂ - ξ₃
    λ2 = ξ₁
    λ3 = ξ₂
    λ4 = ξ₃
    λ = (λ1, λ2, λ3, λ4)
    @inbounds bubble = λ[va] * λ[vb]
    return bubble * w
end

"""
    nedelec_whitney_tet_reference(local_edge::Int, ξ::Vec{3}) -> Vec{3}

Whitney edge basis on the reference `Tet4`, `ξ` in Cartesian coordinates of the
reference simplex (`ξ₁+ξ₂+ξ₃ ≤ 1`, `ξₖ ≥ 0`).
"""
@inline function nedelec_whitney_tet_reference(local_edge::Int, ξ::Vec{3})
    ξ₁ = ξ[1]
    ξ₂ = ξ[2]
    ξ₃ = ξ[3]
    λ1 = 1 - ξ₁ - ξ₂ - ξ₃
    λ2 = ξ₁
    λ3 = ξ₂
    λ4 = ξ₃
    ∇λ1 = Vec((-1.0, -1.0, -1.0))
    ∇λ2 = Vec((1.0, 0.0, 0.0))
    ∇λ3 = Vec((0.0, 1.0, 0.0))
    ∇λ4 = Vec((0.0, 0.0, 1.0))
    λ = (λ1, λ2, λ3, λ4)
    ∇λ = (∇λ1, ∇λ2, ∇λ3, ∇λ4)
    ed = edges(Tet4())[local_edge]
    va, vb = ed.vertices
    @inbounds return λ[va] * ∇λ[vb] - λ[vb] * ∇λ[va]
end
