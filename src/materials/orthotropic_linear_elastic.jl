# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

"""
Orthotropic linear elasticity with principal material axes aligned to global `(x,y,z)`.

Engineering constants `(E₁,E₂,E₃,G₁₂,G₂₃,G₃₁,ν₁₂,ν₂₃,ν₃₁)` follow the usual reciprocal
relations `νᵢⱼ/Eᵢ = νⱼᵢ/Eⱼ`. The compliance uses tensor shear strains on the shear diagonal
(`γᵢⱼ = 2εᵢⱼ` relates work‑conjugate pairs). Stiffness is assembled in Mandel form and mapped
to [`SymmetricTensor`](@ref)`{4,3}` so [`ContinuumKernel`](@ref) and existing assemblers work
unchanged.

This is the same stiffness structure used in composite solid benchmarks such as Code_Aster
orthotropic elasticity documentation (manual **U** / validation decks listing orthotropic
solids — see https://www.code-aster.org/V2/doc/default/en/index.php?man=U ).
"""

using LinearAlgebra
using StaticArrays
using Tensors

const _ORTH_MANDEL_S2 = sqrt(2.0)

@inline function _orthotropic_mandel_second_bases()
    e1 = basevec(Vec{3,Float64}, 1)
    e2 = basevec(Vec{3,Float64}, 2)
    e3 = basevec(Vec{3,Float64}, 3)
    U1 = symmetric(e1 ⊗ e1)
    U2 = symmetric(e2 ⊗ e2)
    U3 = symmetric(e3 ⊗ e3)
    U4 = symmetric((e2 ⊗ e3 + e3 ⊗ e2) / _ORTH_MANDEL_S2)
    U5 = symmetric((e1 ⊗ e3 + e3 ⊗ e1) / _ORTH_MANDEL_S2)
    U6 = symmetric((e1 ⊗ e2 + e2 ⊗ e1) / _ORTH_MANDEL_S2)
    return (U1, U2, U3, U4, U5, U6)
end

"""`𝔻 = Σᵢⱼ Cᵐᵢⱼ Uᵢ ⊗ Uⱼ` with orthonormal Mandel bases `U`."""
function _fourth_order_from_mandel(Cm::SMatrix{6,6,Float64,36})
    Us = _orthotropic_mandel_second_bases()
    𝔻 = zero(SymmetricTensor{4,3,Float64})
    @inbounds for j in 1:6, i in 1:6
        cij = Cm[i, j]
        iszero(cij) && continue
        𝔻 += cij * (Us[i] ⊗ Us[j])
    end
    return 𝔻
end

"""
    OrthotropicLinearElastic <: AbstractElasticMaterial

Nine-parameter orthotropic Hooke solid aligned with global Cartesian axes.

# Constructor

    OrthotropicLinearElastic(; E1, E2, E3, G12, G23, G31, ν12, ν23, ν31)

Reciprocal shear Poisson pairs are filled automatically (`ν₂₁ = ν₁₂ E₂/E₁`, …).

Throws if the compliance matrix is singular or not positive definite.
"""
struct OrthotropicLinearElastic <: AbstractElasticMaterial
    E1::Float64
    E2::Float64
    E3::Float64
    G12::Float64
    G23::Float64
    G31::Float64
    ν12::Float64
    ν23::Float64
    ν31::Float64
    𝔻::SymmetricTensor{4,3,Float64,36}
end

function OrthotropicLinearElastic(;
        E1::Real, E2::Real, E3::Real,
        G12::Real, G23::Real, G31::Real,
        ν12::Real, ν23::Real, ν31::Real,
    )
    E1f = Float64(E1)
    E2f = Float64(E2)
    E3f = Float64(E3)
    G12f = Float64(G12)
    G23f = Float64(G23)
    G31f = Float64(G31)
    ν12f = Float64(ν12)
    ν23f = Float64(ν23)
    ν31f = Float64(ν31)

    E1f > 0 || throw(ArgumentError("E1 must be positive, got E1 = $E1f"))
    E2f > 0 || throw(ArgumentError("E2 must be positive, got E2 = $E2f"))
    E3f > 0 || throw(ArgumentError("E3 must be positive, got E3 = $E3f"))
    G12f > 0 || throw(ArgumentError("G12 must be positive, got G12 = $G12f"))
    G23f > 0 || throw(ArgumentError("G23 must be positive, got G23 = $G23f"))
    G31f > 0 || throw(ArgumentError("G31 must be positive, got G31 = $G31f"))

    ν21 = ν12f * E2f / E1f
    ν32 = ν23f * E3f / E2f
    ν13 = ν31f * E1f / E3f

    S = zeros(Float64, 6, 6)
    S[1, 1] = 1 / E1f
    S[2, 2] = 1 / E2f
    S[3, 3] = 1 / E3f
    S[1, 2] = S[2, 1] = -ν12f / E1f
    S[2, 3] = S[3, 2] = -ν23f / E2f
    S[1, 3] = S[3, 1] = -ν31f / E3f

    S[4, 4] = 1 / G23f
    S[5, 5] = 1 / G31f
    S[6, 6] = 1 / G12f

    Ssym = Symmetric(S)
    λmin = minimum(eigen(Ssym).values)
    λmin > 0 || throw(ArgumentError(
        "Orthotropic compliance must be SPD (smallest eigenvalue = $λmin)"
    ))

    Ceng = inv(Ssym)
    p = @SVector Float64[1.0, 1.0, 1.0, _ORTH_MANDEL_S2, _ORTH_MANDEL_S2, _ORTH_MANDEL_S2]
    # Column-major flat layout for `SMatrix{6,6}`
    Cm = SMatrix{6,6,Float64,36}(ntuple(k -> begin
            j = (k - 1) ÷ 6 + 1
            i = (k - 1) % 6 + 1
            Ceng[i, j] * p[i] * p[j]
        end, 36))

    𝔻 = _fourth_order_from_mandel(Cm)
    return OrthotropicLinearElastic(
        E1f, E2f, E3f, G12f, G23f, G31f, ν12f, ν23f, ν31f, 𝔻,
    )
end

material_behavior(::OrthotropicLinearElastic) = StatelessConstantTangent()
supported_physics(::OrthotropicLinearElastic) = (Elasticity{3}(),)
required_state_variables(::OrthotropicLinearElastic) = ()

function compute_stress(
    material::OrthotropicLinearElastic,
    ε::SymmetricTensor{2,3,T},
    state_old::Union{Nothing,NamedTuple},
    Δt::Float64,
) where T
    σ = dcontract(material.𝔻, ε)
    return σ, material.𝔻, NamedTuple()
end

compute_stress(material::OrthotropicLinearElastic, ε::SymmetricTensor{2,3,T}) where T =
    compute_stress(material, ε, nothing, 0.0)

function elasticity_tensor(material::OrthotropicLinearElastic)
    return material.𝔻
end
