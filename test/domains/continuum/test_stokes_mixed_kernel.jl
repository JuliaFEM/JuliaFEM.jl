# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
`StokesMixedKernel`: Newtonian Stokes u–p on the DOF-based assembler.

1. `K_up`, `K_pu`, `K_pp` match `MixedUPKernel` on the same mesh (geometry-only blocks).
2. `K_uu` scales linearly with viscosity `μ`.
3. Global `K` is numerically symmetric; `assemble!` is allocation-free after warmup.
"""

using Test
using JuliaFEM
using JuliaFEM: StokesMixedKernel, MixedUPKernel, ContinuumFormulation, FullThreeD, LinearElastic
using JuliaFEM: DOFBasedCOOAssembler, DOFBasedCOOCache, assemble!, extract_system
using JuliaFEM: create_unit_cube_mesh, create_elements!, @DOFSet, DOF, Displacement, Vertex
using LinearAlgebra

function _hex8_unit_cube_single_element()
    return create_unit_cube_mesh(Hex8; nx = 1, ny = 1, nz = 1)
end

@testset "StokesMixedKernel vs MixedUPKernel (pressure blocks)" begin
    mesh = _hex8_unit_cube_single_element()
    mat = LinearElastic(E = 210e9, ν = 0.3)
    inv_bulk = 1.0 / (210e9 / 3)

    S = @DOFSet{u::DOF{Displacement{3}, Vertex}, p::DOF{Float64, Cell}}
    elem, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    asm = DOFBasedCOOAssembler()

    k_stokes = StokesMixedKernel(ContinuumFormulation{FullThreeD}(); μ = 1.0e-3, inv_bulk = inv_bulk)
    c_s = DOFBasedCOOCache(elem, handler, mesh, k_stokes)
    assemble!(c_s, asm, k_stokes, mesh)
    K_s, _ = extract_system(c_s)

    k_up = MixedUPKernel(ContinuumFormulation{FullThreeD}(), mat; inv_bulk = inv_bulk)
    c_u = DOFBasedCOOCache(elem, handler, mesh, k_up)
    assemble!(c_u, asm, k_up, mesh)
    K_u, _ = extract_system(c_u)

    n_u = 3 * length(mesh.nodes)
    ru = 1:n_u
    rp = (n_u + 1):(n_u + 1)

    @test norm(K_s[ru, rp] - K_u[ru, rp]) < 1e-12 * max(1.0, norm(K_u[ru, rp]))
    @test norm(K_s[rp, ru] - K_u[rp, ru]) < 1e-12 * max(1.0, norm(K_u[rp, ru]))
    @test abs(K_s[rp[1], rp[1]] - K_u[rp[1], rp[1]]) < 1e-12 * max(1.0, abs(K_u[rp[1], rp[1]]))
end

@testset "StokesMixedKernel K_uu scales with μ" begin
    mesh = _hex8_unit_cube_single_element()
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}, p::DOF{Float64, Cell}}
    elem, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    asm = DOFBasedCOOAssembler()

    k1 = StokesMixedKernel(ContinuumFormulation{FullThreeD}(); μ = 1.0, inv_bulk = 0.0)
    c1 = DOFBasedCOOCache(elem, handler, mesh, k1)
    assemble!(c1, asm, k1, mesh)
    K1, _ = extract_system(c1)

    k2 = StokesMixedKernel(ContinuumFormulation{FullThreeD}(); μ = 2.0, inv_bulk = 0.0)
    c2 = DOFBasedCOOCache(elem, handler, mesh, k2)
    assemble!(c2, asm, k2, mesh)
    K2, _ = extract_system(c2)

    n_u = 3 * length(mesh.nodes)
    ru = 1:n_u
    rel = norm(K2[ru, ru] - 2 * K1[ru, ru]) / max(norm(K1[ru, ru]), 1.0)
    @test rel < 1e-12
end

@testset "StokesMixedKernel symmetry + zero allocations" begin
    mesh = _hex8_unit_cube_single_element()
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}, p::DOF{Float64, Cell}}
    elem, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    k = StokesMixedKernel(ContinuumFormulation{FullThreeD}(); μ = 1.0e-2, inv_bulk = 1e-6)
    asm = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elem, handler, mesh, k)
    assemble!(cache, asm, k, mesh)
    K, _ = extract_system(cache)
    R = K - transpose(K)
    @test norm(R) <= 1e-8 * max(1.0, norm(K))

    for _ in 1:3
        assemble!(cache, asm, k, mesh)
    end
    GC.gc()
    @test (@allocated assemble!(cache, asm, k, mesh)) == 0
end

@testset "StokesMixedKernel API guards" begin
    k = StokesMixedKernel(ContinuumFormulation{FullThreeD}(); μ = 1.0)
    @test dofs_per_node(k) == 4
    @test_throws ErrorException get_field(k)
    @test_throws ArgumentError StokesMixedKernel(ContinuumFormulation{FullThreeD}(); μ = 0.0)
end
