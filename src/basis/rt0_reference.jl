# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Lowest-order **Raviart–Thomas** (`RT₀`) **reference** vector fields for linear
[`Tet4`](@ref) (Whitney-type face fields on the reference simplex) and linear
[`Hex8`](@ref) (tensor-product face fields on the reference cube `[-1,1]³`).

Push-forward to physical space uses [`piola_contravariant`](@ref).

See [`rt0_tet_reference_basis`](@ref), [`rt0_hex8_reference_basis`](@ref).
"""

using Tensors

const _RT0_TET_REF_VOL = 1 / 6

"""
    rt0_tet_reference_basis(local_face::Int, ξ::Vec{3}) -> Vec{3}

`RT₀`-type reference vector field for triangular face `local_face ∈ 1:4`,
evaluated at reference coordinates `ξ = (ξ₁,ξ₂,ξ₃)` (barycentric complement
`λ₁ = 1 - ξ₁ - ξ₂ - ξ₃ ≥ 0`).

Uses the cyclic Whitney face combination on the three vertices of that face:

``\\mathbf{\\psi}_F = \\dfrac{2}{|T|}\\sum_{(i,j,k)\\ \\mathrm{cyclic}}
    \\lambda_i\\,(\\nabla\\lambda_j\\times\\nabla\\lambda_k)``

with `|T| = 1/6` on the reference tet.
"""
@inline function rt0_tet_reference_basis(local_face::Int, ξ::Vec{3})
    λ1 = 1 - ξ[1] - ξ[2] - ξ[3]
    λ = (λ1, ξ[1], ξ[2], ξ[3])
    ∇λ1 = Vec((-1.0, -1.0, -1.0))
    ∇λ2 = Vec((1.0, 0.0, 0.0))
    ∇λ3 = Vec((0.0, 1.0, 0.0))
    ∇λ4 = Vec((0.0, 0.0, 1.0))
    ∇λ = (∇λ1, ∇λ2, ∇λ3, ∇λ4)
    fc = faces(Tet4())[local_face]
    a, b, c = fc.vertices
    scale = 2 / _RT0_TET_REF_VOL
    @inbounds return scale * (
        λ[a] * (∇λ[b] × ∇λ[c]) +
        λ[b] * (∇λ[c] × ∇λ[a]) +
        λ[c] * (∇λ[a] × ∇λ[b])
    )
end

"""
    rt0_hex8_reference_basis(local_face::Int, ξ::Vec{3}) -> Vec{3}

`RT₀`-type reference vector field for quadrilateral face `local_face ∈ 1:6`,
evaluated at reference coordinates `ξ = (ξ₁, ξ₂, ξ₃)` on the [`Hex8`](@ref)
reference cube `[-1,1]³` ([`reference_coordinates(::Hexahedron{8})`](@ref)).

Ordering matches [`faces(::Hexahedron)`](@ref): faces are the bottom/top,
front/back, left/right caps in that listing; each basis has **unit outward
normal flux** on its face, **zero** flux through the opposite face, and
**constant divergence** `½` on the reference cell.

Component pattern (with `ξ[k]` the reference coordinates on `[-1,1]`):

| `local_face` | nonzero component | formula |
| :--- | :--- | :--- |
| 1 (`z=-`) | `z` | `-(1-ξ₃)/2` |
| 2 (`z=+`) | `z` | `(1+ξ₃)/2` |
| 3 (`y=-`) | `y` | `-(1-ξ₂)/2` |
| 4 (`x=+`) | `x` | `(1+ξ₁)/2` |
| 5 (`y=+`) | `y` | `(1+ξ₂)/2` |
| 6 (`x=-`) | `x` | `-(1-ξ₁)/2` |
"""
@inline function rt0_hex8_reference_basis(local_face::Int, ξ::Vec{3})
    ξ1 = ξ[1]
    ξ2 = ξ[2]
    ξ3 = ξ[3]
    if local_face == 1
        return Vec((0.0, 0.0, -(1 - ξ3) / 2))
    elseif local_face == 2
        return Vec((0.0, 0.0, (1 + ξ3) / 2))
    elseif local_face == 3
        return Vec((0.0, -(1 - ξ2) / 2, 0.0))
    elseif local_face == 4
        return Vec(((1 + ξ1) / 2, 0.0, 0.0))
    elseif local_face == 5
        return Vec((0.0, (1 + ξ2) / 2, 0.0))
    elseif local_face == 6
        return Vec((-(1 - ξ1) / 2, 0.0, 0.0))
    else
        throw(ArgumentError("local_face must be in 1:6 (Hex8 has $(nfaces(Hex8))) faces, got $local_face"))
    end
end
