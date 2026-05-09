# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using Test
using JuliaFEM
using Tensors

@testset "Piola identity on J = I" begin
    J = Tensor{2, 3}((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    u = Vec((1.0, -2.0, 0.5))
    @test piola_covariant(J, u) ≈ u
    @test piola_contravariant(J, u) ≈ u
end

@testset "Nédélec Whitney tet — tangential component on edge (1,2)" begin
    ξ = Vec((0.3, 0.0, 0.0))
    N = nedelec_whitney_tet_reference(1, ξ)
    ex = Vec((1.0, 0.0, 0.0))
    @test dot(N, ex) ≈ 1.0 rtol = 1e-14
end

@testset "Piola covariant commutes with identity Jacobian (reference point)" begin
    # Physical tet = reference tet: Jacobian is identity at any ξ for affine map.
    J = Tensor{2, 3}((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    ξ = Vec((0.2, 0.2, 0.2))
    u_ref = nedelec_whitney_tet_reference(4, ξ)
    @test piola_covariant(J, u_ref) ≈ u_ref
end

@testset "Nédélec hierarchical tet — slot 1 matches Whitney" begin
    ξ = Vec((0.17, 0.09, 0.05))
    for le in 1:6
        @test nedelec_hierarchical_tet_edge_reference(le, 1, ξ) ≈ nedelec_whitney_tet_reference(le, ξ)
    end
end

@testset "Nédélec hierarchical tet — slot 2 tangential on edge (1,2)" begin
    ξ = Vec((0.3, 0.0, 0.0))
    ex = Vec((1.0, 0.0, 0.0))
    N2 = nedelec_hierarchical_tet_edge_reference(1, 2, ξ)
    @test dot(N2, ex) ≈ 0.7 * 0.3 rtol = 1e-14
end

@testset "Nédélec hierarchical tet — Piola covariant with diagonal Jacobian" begin
    J = Tensor{2, 3}((2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0))
    ξ = Vec((0.11, 0.22, 0.07))
    le = 3
    for slot in 1:2
        u_ref = nedelec_hierarchical_tet_edge_reference(le, slot, ξ)
        u_phys = piola_covariant(J, u_ref)
        inv_diag = Vec((1 / 2, 1 / 3, 1 / 4))
        @test u_phys ≈ inv_diag .* u_ref rtol = 1e-14
    end
end

@testset "Piola contravariant with diagonal non-unit Jacobian" begin
    J = Tensor{2, 3}((2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0))
    ex = Vec((1.0, 0.0, 0.0))
    @test piola_contravariant(J, ex) ≈ Vec((1.0 / 12.0, 0.0, 0.0))
end

# Reference ξ; physical `x` diagonal with `∂/∂x_i = (1/J_ii) ∂/∂ξ_i`.
function _divergence_physical_diagonal(f, ξ::Vec{3}, inv_diag::Vec{3}; h::Float64 = 1e-6)
    e1 = Vec((1.0, 0.0, 0.0))
    e2 = Vec((0.0, 1.0, 0.0))
    e3 = Vec((0.0, 0.0, 1.0))
    return (
        inv_diag[1] * (f(ξ + h * e1)[1] - f(ξ - h * e1)[1]) / (2h) +
        inv_diag[2] * (f(ξ + h * e2)[2] - f(ξ - h * e2)[2]) / (2h) +
        inv_diag[3] * (f(ξ + h * e3)[3] - f(ξ - h * e3)[3]) / (2h)
    )
end

function _divergence_reference(f, ξ::Vec{3}; h::Float64 = 1e-6)
    e1 = Vec((1.0, 0.0, 0.0))
    e2 = Vec((0.0, 1.0, 0.0))
    e3 = Vec((0.0, 0.0, 1.0))
    return (
        (f(ξ + h * e1)[1] - f(ξ - h * e1)[1]) / (2h) +
        (f(ξ + h * e2)[2] - f(ξ - h * e2)[2]) / (2h) +
        (f(ξ + h * e3)[3] - f(ξ - h * e3)[3]) / (2h)
    )
end

@testset "RT₀ tet reference — divergence constant (finite difference)" begin
    ξa = Vec((0.11, 0.22, 0.05))
    ξb = Vec((0.05, 0.07, 0.08))
    for lf in 1:4
        div_a = _divergence_reference(ξ -> rt0_tet_reference_basis(lf, ξ), ξa)
        div_b = _divergence_reference(ξ -> rt0_tet_reference_basis(lf, ξ), ξb)
        @test div_a ≈ div_b rtol = 1e-4
        @test div_a ≈ 36.0 rtol = 1e-5
    end
end

# Outward face normals for Hex8 reference faces (same order as `faces(::Hex8)`).
const _RT0_HEX_FACE_NORMALS = (
    Vec((0.0, 0.0, -1.0)), # z = -1
    Vec((0.0, 0.0, 1.0)),  # z = +1
    Vec((0.0, -1.0, 0.0)), # y = -1
    Vec((1.0, 0.0, 0.0)),  # x = +1
    Vec((0.0, 1.0, 0.0)),  # y = +1
    Vec((-1.0, 0.0, 0.0)), # x = -1
)

@testset "RT₀ hex reference — divergence constant (finite difference)" begin
    ξa = Vec((0.11, -0.22, 0.31))
    ξb = Vec((-0.05, 0.17, -0.08))
    for lf in 1:6
        div_a = _divergence_reference(ξ -> rt0_hex8_reference_basis(lf, ξ), ξa)
        div_b = _divergence_reference(ξ -> rt0_hex8_reference_basis(lf, ξ), ξb)
        @test div_a ≈ div_b rtol = 1e-5
        @test div_a ≈ 0.5 rtol = 1e-5
    end
end

@testset "RT₀ hex reference — unit outward normal flux on owning face" begin
    face_points = (
        Vec((0.25, 0.4, -1.0)),
        Vec((-0.5, -0.2, 1.0)),
        Vec((0.2, -1.0, 0.35)),
        Vec((1.0, -0.1, 0.2)),
        Vec((-0.3, 1.0, -0.4)),
        Vec((-1.0, 0.15, 0.25)),
    )
    for lf in 1:6
        ξ = face_points[lf]
        ψ = rt0_hex8_reference_basis(lf, ξ)
        n = _RT0_HEX_FACE_NORMALS[lf]
        @test dot(ψ, n) ≈ 1.0 rtol = 1e-14
    end
end

@testset "RT₀ hex + Piola contravariant — divergence scales by 1/det(J)" begin
    J = Tensor{2, 3}((2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0))
    inv_diag = Vec((1 / 2, 1 / 3, 1 / 4))
    ξ = Vec((0.15, -0.25, 0.05))
    lf = 4
    div_ref = _divergence_reference(ξ -> rt0_hex8_reference_basis(lf, ξ), ξ)
    div_phys = _divergence_physical_diagonal(
        ξ -> piola_contravariant(J, rt0_hex8_reference_basis(lf, ξ)),
        ξ,
        inv_diag,
    )
    @test div_ref ≈ 0.5 rtol = 1e-5
    @test div_phys ≈ div_ref / 24 rtol = 1e-4
end

@testset "RT₀ hex reference — zero normal flux on opposite face" begin
    # Parallel face pairs: (1,2), (3,5), (4,6) in `faces(::Hex8)` order.
    lf_opposite = (2, 1, 5, 6, 3, 4)
    opposite_points = (
        Vec((0.1, 0.2, 1.0)),   # on face 2 (z=+)
        Vec((0.2, -0.3, -1.0)), # on face 1 (z=-)
        Vec((-0.2, 1.0, 0.1)),  # on face 5 (y=+)
        Vec((-1.0, 0.3, -0.1)), # on face 6 (x=-)
        Vec((0.4, -1.0, 0.2)),  # on face 3 (y=-)
        Vec((1.0, -0.2, 0.3)),  # on face 4 (x=+)
    )
    for lf in 1:6
        ξ = opposite_points[lf]
        ψ = rt0_hex8_reference_basis(lf, ξ)
        n_opp = _RT0_HEX_FACE_NORMALS[lf_opposite[lf]]
        @test dot(ψ, n_opp) ≈ 0.0 atol = 1e-14
    end
end
