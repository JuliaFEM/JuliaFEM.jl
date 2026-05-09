# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

using Test
using JuliaFEM
using Tensors

@testset "J2LinearIsotropicPlasticity" begin
    @testset "traits and state layout (no backstress)" begin
        mat = J2LinearIsotropicPlasticity(E = 210e9, ν = 0.3, σ_y0 = 250e6, H_iso = 1e9)
        @test material_behavior(mat) isa StatefulStrainDependent
        @test required_state_variables(mat) === (PlasticStrain, EquivalentPlasticStrain)
        _, _, st = compute_stress(mat, zero(SymmetricTensor{2,3}), NamedTuple(), 0.0)
        @test !haskey(st, :α)
    end

    @testset "von Mises stress on yield surface" begin
        mat = J2LinearIsotropicPlasticity(E = 210e9, ν = 0.3, σ_y0 = 250e6, H_iso = 2e9)
        ε = SymmetricTensor{2,3}((0.008, 0.0, 0.0, 0.0, 0.0, 0.0))
        σ, _, st = compute_stress(mat, ε, NamedTuple(), 0.0)
        s = dev(σ)
        seq = √(3 / 2) * √(s ⊡ s)
        σ_y = mat.σ_y0 + mat.H_iso * st.κ
        @test seq ≈ σ_y rtol = 1e-6
    end

    @testset "distinct from kinematic PerfectPlasticity (stress path)" begin
        E = 210e9
        ν = 0.3
        σ_y = 250e6
        H = 1e9
        ε_load = SymmetricTensor{2,3}((0.006, 0.0, 0.0, 0.0, 0.0, 0.0))

        mk = PerfectPlasticity(E = E, ν = ν, σ_y = σ_y, H = H)
        mi = J2LinearIsotropicPlasticity(E = E, ν = ν, σ_y0 = σ_y, H_iso = H)

        σk, _, _ = compute_stress(mk, ε_load, NamedTuple(), 0.0)
        σi, _, _ = compute_stress(mi, ε_load, NamedTuple(), 0.0)
        @test σk ≠ σi
    end
end
