# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using Test
using LinearAlgebra
using SparseArrays
using JuliaFEM
using JuliaFEM: MixedUPKernel, ContinuumFormulation, FullThreeD
using JuliaFEM: DOFBasedCOOAssembler, DOFBasedCOOCache, assemble!, extract_system
using JuliaFEM: create_unit_cube_mesh, create_elements!, @DOFSet, DOF, Displacement, Vertex, Cell
using JuliaFEM: global_field_ranges, saddle_point_matrix_blocks,
    ApproxSchurDiagBlockPreconditioner, ApproxSchurICholDiagBlockPreconditioner

@testset "saddle_point_matrix_blocks + ApproxSchurDiagBlockPreconditioner (3×3 model)" begin
    Kd = [2.0 0.0 1.0; 0.0 3.0 2.0; 1.0 2.0 -1.0]
    K = sparse(Kd)
    ru, rp = 1:2, 3:3
    blks = saddle_point_matrix_blocks(K, ru, rp)
    @test Matrix(blks.A) ≈ Kd[1:2, 1:2]
    @test vec(Matrix(blks.B)) ≈ [1.0, 2.0]
    @test blks.C[1, 1] ≈ -1.0

    P = ApproxSchurDiagBlockPreconditioner(K, ru, rp)
    @test P.inv_diag_A ≈ [0.5, 1.0 / 3.0]
    sdiag = -1.0 - (1.0^2) / 2.0 - (2.0^2) / 3.0
    @test P.inv_schur_diag[1] ≈ 1.0 / sdiag

    x = [1.0, -2.0, 0.5]
    y = similar(x)
    ldiv!(y, P, x)
    @test y[1] ≈ 0.5 * 1.0
    @test y[2] ≈ (1.0 / 3.0) * (-2.0)
    @test y[3] ≈ P.inv_schur_diag[1] * 0.5
    @test all(isfinite, y)
end

@testset "ApproxSchurICholDiagBlockPreconditioner (3×3 SPD A block)" begin
    Kd = [2.0 0.0 1.0; 0.0 3.0 2.0; 1.0 2.0 -1.0]
    K = sparse(Kd)
    P = ApproxSchurICholDiagBlockPreconditioner(K, 1:2, 3:3)
    x = [1.0, -2.0, 0.5]
    y = similar(x)
    ldiv!(y, P, x)
    @test y[1:2] ≈ Kd[1:2, 1:2] \ x[1:2]
    sdiag = -1.0 - (1.0^2) / 2.0 - (2.0^2) / 3.0
    @test y[3] ≈ x[3] / sdiag
end

@testset "ApproxSchurDiagBlockPreconditioner — MixedUP Hex8 single element smoke" begin
    mesh = create_unit_cube_mesh(Hex8; nx = 1, ny = 1, nz = 1)
    mat = LinearElastic(E = 210e9, ν = 0.3)
    κ = mat.E / (3 * (1 - 2 * mat.ν))
    S_up = @DOFSet{u::DOF{Displacement{3}, Vertex}, p::DOF{Float64, Cell}}
    elem, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S_up})
    kup = MixedUPKernel(ContinuumFormulation{FullThreeD}(), mat; inv_bulk = 1.0 / κ)
    asm = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elem, handler, mesh, kup)
    assemble!(cache, asm, kup, mesh)
    K, _ = extract_system(cache)
    ru, rp = global_field_ranges(handler)
    P = ApproxSchurDiagBlockPreconditioner(K, ru, rp)
    n = size(K, 1)
    x = randn(n)
    y = zeros(n)
    ldiv!(y, P, x)
    @test all(isfinite, y)
    @test length(P.inv_diag_A) == length(ru)
    @test length(P.inv_schur_diag) == length(rp)
end
