# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

using Test
using JuliaFEM
using Tensors

@testset "update_material_cache! behavior branches" begin
    mesh = create_test_mesh()
    N = 8
    NIP = 8
    geometry_cache = JuliaFEM.create_geometry_cache(N, NIP)
    elem_id = 1
    Δt = 0.0

    @testset "StatelessStrainDependent (NeoHookean)" begin
        mat = NeoHookean(E_mod=210e9, nu=0.3)
        kernel = ContinuumKernel(ContinuumFormulation{FullThreeD}(), mat)
        element_cache = JuliaFEM.create_element_cache(mesh, kernel)
        material_cache = JuliaFEM.create_material_cache(mat, NIP)
        global_cache = JuliaFEM.create_global_material_cache(mat; n_ips=NIP, n_elems=1)

        ug = [zero(Vec{3,Float64}) for _ in 1:nnodes_total(mesh)]
        ug[2] = Vec((0.05, 0.0, 0.0))

        JuliaFEM.update_geometry_cache!(geometry_cache, element_cache, elem_id, mesh)
        JuliaFEM.update_element_cache!(element_cache, kernel, elem_id, mesh, ug)
        JuliaFEM.update_material_cache!(
            material_cache, geometry_cache, mat, element_cache, global_cache, elem_id, Δt,
        )

        σ = JuliaFEM.get_stress(material_cache, 1)
        @test norm(σ) > 1.0
    end

    @testset "StatefulStrainDependent (PerfectPlasticity)" begin
        mat = PerfectPlasticity(; E=210e9, ν=0.3, σ_y=200e6, H=0.0)
        kernel = ContinuumKernel(ContinuumFormulation{FullThreeD}(), mat)
        element_cache = JuliaFEM.create_element_cache(mesh, kernel)
        material_cache = JuliaFEM.create_material_cache(mat, NIP)
        global_cache = JuliaFEM.create_global_material_cache(mat; n_ips=NIP, n_elems=1)

        ug = [zero(Vec{3,Float64}) for _ in 1:nnodes_total(mesh)]
        ug[1] = Vec((0.0, 0.0, 0.0))
        ug[2] = Vec((0.002, 0.0, 0.0))

        JuliaFEM.update_geometry_cache!(geometry_cache, element_cache, elem_id, mesh)
        JuliaFEM.update_element_cache!(element_cache, kernel, elem_id, mesh, ug)
        JuliaFEM.update_material_cache!(
            material_cache, geometry_cache, mat, element_cache, global_cache, elem_id, Δt,
        )

        σ = JuliaFEM.get_stress(material_cache, 1)
        @test σ isa SymmetricTensor{2,3,Float64}
    end
end
