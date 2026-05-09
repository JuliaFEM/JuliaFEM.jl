# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

using Test
using Random
using Tensors
using JuliaFEM
using JuliaFEM: OrthotropicLinearElastic, LinearElastic, compute_stress, elasticity_tensor
using JuliaFEM: material_behavior, StatelessConstantTangent, supported_physics, Elasticity
using JuliaFEM: required_state_variables

@testset "OrthotropicLinearElastic" begin
    @testset "traits match isotropic elastic" begin
        E = 70e9
        ν = 0.33
        G = E / (2(1 + ν))
        mat = OrthotropicLinearElastic(;
            E1 = E, E2 = E, E3 = E, G12 = G, G23 = G, G31 = G, ν12 = ν, ν23 = ν, ν31 = ν,
        )
        @test material_behavior(mat) isa StatelessConstantTangent
        @test supported_physics(mat) == (Elasticity{3}(),)
        @test required_state_variables(mat) == ()
    end

    @testset "isotropic reduction: tangent matches LinearElastic (Mandel)" begin
        E = 210e9
        ν = 0.3
        G = E / (2(1 + ν))
        iso = LinearElastic(E = E, ν = ν)
        ort = OrthotropicLinearElastic(;
            E1 = E, E2 = E, E3 = E, G12 = G, G23 = G, G31 = G, ν12 = ν, ν23 = ν, ν31 = ν,
        )
        ε0 = zero(SymmetricTensor{2,3,Float64})
        _, 𝔻_iso, _ = compute_stress(iso, ε0, NamedTuple(), 0.0)
        𝔻_ort = elasticity_tensor(ort)
        @test tomandel(𝔻_iso) ≈ tomandel(𝔻_ort)
    end

    @testset "isotropic reduction: stress matches LinearElastic for random strain" begin
        E = 50e9
        ν = 0.27
        G = E / (2(1 + ν))
        iso = LinearElastic(E = E, ν = ν)
        ort = OrthotropicLinearElastic(;
            E1 = E, E2 = E, E3 = E, G12 = G, G23 = G, G31 = G, ν12 = ν, ν23 = ν, ν31 = ν,
        )
        rng = MersenneTwister(20260509)
        for _ in 1:12
            v = ntuple(_ -> randn(rng), 6)
            ε = SymmetricTensor{2,3}(v)
            σ_i, _, _ = compute_stress(iso, ε, NamedTuple(), 0.0)
            σ_o, _, _ = compute_stress(ort, ε, NamedTuple(), 0.0)
            @test σ_i ≈ σ_o
        end
    end

    @testset "orthotropic coupling: σ₁ dominates under uniaxial ε₁₁" begin
        mat = OrthotropicLinearElastic(;
            E1 = 200e9,
            E2 = 10e9,
            E3 = 10e9,
            G12 = 5e9,
            G23 = 4e9,
            G31 = 4e9,
            ν12 = 0.03,
            ν23 = 0.35,
            ν31 = 0.03,
        )
        ε = SymmetricTensor{2,3}((1e-3, 0.0, 0.0, 0.0, 0.0, 0.0))
        σ, _, _ = compute_stress(mat, ε, NamedTuple(), 0.0)
        @test abs(σ[1, 1]) > 10 * abs(σ[2, 2])
        @test abs(σ[1, 1]) > 10 * abs(σ[3, 3])
    end

    @testset "invalid compliance throws" begin
        @test_throws ArgumentError OrthotropicLinearElastic(;
            E1 = 1e9, E2 = 1e9, E3 = 1e9,
            G12 = 1e9, G23 = 1e9, G31 = 1e9,
            ν12 = 0.99, ν23 = 0.99, ν31 = 0.99,
        )
    end
end
