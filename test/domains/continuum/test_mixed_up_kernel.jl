# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
`MixedUPKernel`: displacement–pressure (P0) on the DOF-based assembler.

1. `K_uu` matches `ContinuumKernel` on the same mesh when `inv_bulk = 0`
   (`Hex8` / `Lagrange{1}` brick and reference `Tet10` / `Lagrange{2}`).
2. `global_field_ranges` / `saddle_point_blocks` recover the `u` block and couplings.
3. Global `K` is numerically symmetric where tested.
4. `assemble!` allocates 0 bytes after warmup (same contract as other kernels).
"""

using Test
using JuliaFEM
using JuliaFEM: MixedUPKernel, ContinuumKernel, ContinuumFormulation, FullThreeD
using JuliaFEM: DOFBasedCOOAssembler, DOFBasedCOOCache, assemble!, extract_system
using JuliaFEM: create_unit_cube_mesh, create_elements!, @DOFSet, DOF, Displacement, Vertex
using JuliaFEM: global_field_ranges, saddle_point_blocks
using LinearAlgebra
using SparseArrays

function _hex8_unit_cube_single_element()
    return create_unit_cube_mesh(Hex8; nx = 1, ny = 1, nz = 1)
end

@testset "MixedUPKernel vs ContinuumKernel (K_uu block)" begin
    mesh = _hex8_unit_cube_single_element()
    mat = LinearElastic(E = 210e9, ν = 0.3)

    S_u = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    elem_u, h_u = create_elements!(mesh, Element{Hex8, Lagrange{1}, S_u})
    ku = ContinuumKernel(ContinuumFormulation{FullThreeD}(), mat)
    asm = DOFBasedCOOAssembler()
    cache_u = DOFBasedCOOCache(elem_u, h_u, mesh, ku)
    assemble!(cache_u, asm, ku, mesh)
    K_u, _ = extract_system(cache_u)

    S_up = @DOFSet{u::DOF{Displacement{3}, Vertex}, p::DOF{Float64, Cell}}
    elem_up, h_up = create_elements!(mesh, Element{Hex8, Lagrange{1}, S_up})
    kup = MixedUPKernel(ContinuumFormulation{FullThreeD}(), mat; inv_bulk = 0.0)
    cache_up = DOFBasedCOOCache(elem_up, h_up, mesh, kup)
    assemble!(cache_up, asm, kup, mesh)
    K_up, _ = extract_system(cache_up)

    n_u = 3 * length(mesh.nodes)
    @test size(K_up, 1) == n_u + 1
    Kuu = K_up[1:n_u, 1:n_u]

    rel = norm(Kuu - K_u) / max(norm(K_u), 1.0)
    @test rel < 1e-12

    ru, rp = global_field_ranges(h_up)
    @test ru == 1:n_u
    @test rp == (n_u + 1):(n_u + 1)
    blk = saddle_point_blocks(K_up, ru, rp)
    @test blk.A ≈ Kuu
    @test norm(blk.B) > 0.0
    # Symmetric bilinear form for this kernel; general saddle-point Jacobians need not satisfy this.
    @test blk.Bt ≈ transpose(blk.B)

    # Pressure–pressure row/column: single dof, zero compressibility
    @test abs(K_up[n_u + 1, n_u + 1]) < 1e-20
end

@testset "MixedUPKernel global symmetry + K_pp sign" begin
    mesh = _hex8_unit_cube_single_element()
    mat = LinearElastic(E = 210e9, ν = 0.3)
    κ = mat.E / (3 * (1 - 2 * mat.ν))  # isotropic bulk modulus
    inv_bulk = 1.0 / κ

    S_up = @DOFSet{u::DOF{Displacement{3}, Vertex}, p::DOF{Float64, Cell}}
    elem_up, h_up = create_elements!(mesh, Element{Hex8, Lagrange{1}, S_up})
    kup = MixedUPKernel(ContinuumFormulation{FullThreeD}(), mat; inv_bulk = inv_bulk)
    asm = DOFBasedCOOAssembler()
    cache_up = DOFBasedCOOCache(elem_up, h_up, mesh, kup)
    assemble!(cache_up, asm, kup, mesh)
    K, _ = extract_system(cache_up)

    R = K - transpose(K)
    @test norm(R) <= 1e-8 * max(1.0, norm(K))

    n_u = 3 * length(mesh.nodes)
    kpp = K[n_u + 1, n_u + 1]
    @test kpp < 0.0
end

@testset "MixedUPKernel Tet10 quadratic — K_uu vs ContinuumKernel" begin
    nodes = Vec{3, Float64}[reference_coordinates(Tet10())...]
    conn = ntuple(i -> UInt32(i), 10)
    mesh = Mesh{Tet10}(nodes, [conn])
    mat = LinearElastic(E = 210e9, ν = 0.3)

    S_u = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    elem_u, h_u = create_elements!(mesh, Element{Tet10, Lagrange{2}, S_u})
    ku = ContinuumKernel(ContinuumFormulation{FullThreeD}(), mat)
    asm = DOFBasedCOOAssembler()
    cache_u = DOFBasedCOOCache(elem_u, h_u, mesh, ku)
    assemble!(cache_u, asm, ku, mesh)
    K_u, _ = extract_system(cache_u)

    S_up = @DOFSet{u::DOF{Displacement{3}, Vertex}, p::DOF{Float64, Cell}}
    elem_up, h_up = create_elements!(mesh, Element{Tet10, Lagrange{2}, S_up})
    kup = MixedUPKernel(ContinuumFormulation{FullThreeD}(), mat; inv_bulk = 0.0)
    cache_up = DOFBasedCOOCache(elem_up, h_up, mesh, kup)
    assemble!(cache_up, asm, kup, mesh)
    K_up, _ = extract_system(cache_up)

    ru, rp = global_field_ranges(h_up)
    n_u = 3 * length(mesh.nodes)
    @test length(ru) == n_u
    @test length(rp) == 1
    blk = saddle_point_blocks(K_up, ru, rp)
    @test size(K_u, 1) == n_u
    rel = norm(Matrix(blk.A) - Matrix(K_u)) / max(norm(K_u), 1.0)
    @test rel < 1e-11

    R = K_up - transpose(K_up)
    @test norm(R) <= 1e-8 * max(1.0, norm(K_up))
end

@testset "MixedUPKernel assemble! zero allocations" begin
    mesh = _hex8_unit_cube_single_element()
    mat = LinearElastic(E = 210e9, ν = 0.3)
    S_up = @DOFSet{u::DOF{Displacement{3}, Vertex}, p::DOF{Float64, Cell}}
    elem_up, h_up = create_elements!(mesh, Element{Hex8, Lagrange{1}, S_up})
    kup = MixedUPKernel(ContinuumFormulation{FullThreeD}(), mat; inv_bulk = 1e-11)
    asm = DOFBasedCOOAssembler()
    cache_up = DOFBasedCOOCache(elem_up, h_up, mesh, kup)
    for _ in 1:3
        assemble!(cache_up, asm, kup, mesh)
    end
    GC.gc()
    @test (@allocated assemble!(cache_up, asm, kup, mesh)) == 0
end

@testset "MixedUPKernel API guards" begin
    mat = LinearElastic(E = 210e9, ν = 0.3)
    k = MixedUPKernel(ContinuumFormulation{FullThreeD}(), mat)
    @test dofs_per_node(k) == 4
    @test_throws ErrorException get_field(k)
end
