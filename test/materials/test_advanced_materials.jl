# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

using Test
using JuliaFEM
using Tensors

@testset "Advanced materials (hyperelastic, plasticity variants, diffusion)" begin
    @testset "continuum_kinematics defaults and StVK-J2" begin
        @test continuum_kinematics(LinearElastic(E = 210e9, ν = 0.3)) isa SmallStrainKinematics
        stvk = StVenantKirchhoffJ2Plasticity(E = 210e9, ν = 0.3, σ_y = 300e6, H = 0.0)
        @test continuum_kinematics(stvk) isa GreenLagrangeKinematics
        Egl = SymmetricTensor{2,3}((Float64(π) * 1e-7, 2e-7, -3e-7, 4e-7, 5e-7, -1e-7))
        le = LinearElastic(E = 210e9, ν = 0.3)
        σ_stvk, _, _ = compute_stress(stvk, Egl, NamedTuple(), 0.0)
        σ_le, _, _ = compute_stress(le, Egl, NamedTuple(), 0.0)
        @test σ_stvk ≈ σ_le rtol = 1e-6 atol = 1.0
        σ_nothing, _, _ = compute_stress(stvk, Egl, nothing, 0.0)
        @test σ_nothing ≈ σ_stvk
    end

    @testset "Hyperelastic models (Nothing / NamedTuple state)" begin
        E0 = zero(SymmetricTensor{2,3})
        for mat in (
            MooneyRivlin(C10 = 80e3, C01 = 20e3, κ_bulk = 1e9),
            Yeoh3(C10 = 1e5, C20 = -500.0, C30 = 10.0, κ_bulk = 1e9),
            Gent(μ = 1e5, Jm = 100.0, κ_bulk = 1e9),
        )
            S1, 𝔻1, st1 = compute_stress(mat, E0, nothing, 0.0)
            S2, 𝔻2, st2 = compute_stress(mat, E0, NamedTuple(), 0.0)
            @test S1 isa SymmetricTensor{2,3}
            @test S1 ≈ S2
            @test 𝔻1 isa SymmetricTensor{4,3}
            @test st1 isa NamedTuple
            @test st2 isa NamedTuple
        end
    end

    @testset "Scalar damage" begin
        mat = ScalarDamageLinearElastic(E = 210e9, ν = 0.3, r = 100.0, ε0 = 0.0)
        ε = 0.001 * one(SymmetricTensor{2,3})
        σ, _, st = compute_stress(mat, ε, nothing, 0.0)
        @test st.d ≥ 0.0
        @test st.κ_d ≥ 0.0
        @test σ isa SymmetricTensor{2,3}
    end

    @testset "Chaboche J2 (elastic branch)" begin
        m = ChabocheJ2Plasticity(
            E = 210e9, ν = 0.3, σ_y = 400e6,
            C1 = 50e9, γ1 = 100.0, C2 = 30e9, γ2 = 50.0,
        )
        ε = 1e-8 * one(SymmetricTensor{2,3})
        σ, _, st = compute_stress(m, ε, nothing, 0.0)
        @test st.κ == 0.0
        @test σ isa SymmetricTensor{2,3}
    end

    @testset "Norton creep elastic + volumetric swelling" begin
        mat = NortonCreepElastic(E = 200e9, ν = 0.3, A = 1e-22, n = 3.0, β_swelling = 1e-4, phi_dot = 2.0)
        ε = 0.002 * one(SymmetricTensor{2,3})
        σ, _, st = compute_stress(mat, ε, NamedTuple(), 1.0)
        @test st.ε_c isa SymmetricTensor{2,3}
        @test tr(st.ε_c) > 0.0
        @test σ isa SymmetricTensor{2,3}
    end

    @testset "Linear elastic with eigenstrain" begin
        mat = LinearElasticWithEigenstrain(E = 210e9, ν = 0.3)
        ε_e = 1e-4 * one(SymmetricTensor{2,3})
        σ, _, st = compute_stress(mat, ε_e, (ε_e = ε_e,), 0.0)
        @test norm(σ) ≤ 1e3
        @test st.ε_e ≈ ε_e
    end

    @testset "Moisture diffusion kernel wiring" begin
        D = MoistureDiffusivity(D_w = 2.5e-9)
        @test tr(conductivity_tensor(D)) > 0.0
        kernel = HeatKernel(ContinuumFormulation{FullThreeD}(), D)
        @test get_field(kernel) isa MoistureContent
        rf = reference_fields(kernel)
        @test rf[1].k ≈ scalar_diffusion_tensor(D)
        @test_throws ArgumentError HeatKernel(
            ContinuumFormulation{FullThreeD}(),
            MoistureDiffusivity(D_w = 1.0),
            Temperature(),
        )
        @test_throws ArgumentError HeatKernel(
            ContinuumFormulation{FullThreeD}(),
            HeatConductivity(k = 1.0),
            MoistureContent(),
        )
    end
end
