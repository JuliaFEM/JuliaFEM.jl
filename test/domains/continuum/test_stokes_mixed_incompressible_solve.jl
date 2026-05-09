# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
`StokesMixedKernel`: same brick BC/load pattern as incompressible `MixedUPKernel`,
direct sparse solve, then matrix-free vs assembled `K` and Jacobi diagonal check.
"""

using Test
using JuliaFEM
using JuliaFEM: StokesMixedKernel, ContinuumFormulation, FullThreeD
using JuliaFEM: DOFBasedCOOAssembler, DOFBasedCOOCache, assemble!, extract_system
using JuliaFEM: default_pressure_gauge_dof, matrix_free_op
using JuliaFEM: PenaltyDirichlet, apply_constraint!, apply_load!
using JuliaFEM: compute_diagonal!, apply_constraint_diag!
using JuliaFEM: NodalForce
using JuliaFEM: @DOFSet, DOF, Displacement, Vertex, Hex8
using LinearAlgebra
using SparseArrays

include("solve_brick_helpers.jl")

@testset "StokesMixedKernel: brick direct solve + pressure gauge" begin
    mesh = brick_hex_mesh(2, 2, 2)
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}, p::DOF{Float64, Cell}}
    elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    kernel = StokesMixedKernel(ContinuumFormulation{FullThreeD}(); μ = 1.0e-3, inv_bulk = 0.0)
    asm = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elements, handler, mesh, kernel)
    assemble!(cache, asm, kernel, mesh)
    K, f = extract_system(cache)
    fill!(f, 0.0)

    u_fix = collect_vertex_dofs_on_nodeset(handler, mesh, :xmin)
    p_pin = default_pressure_gauge_dof(handler; elem_id = 1)
    fixed = Int[vcat(u_fix, [p_pin])...]
    λ = 1.0e14
    bc = PenaltyDirichlet(fixed, zeros(Float64, length(fixed)); penalty = λ)

    uz = interior_uz_dof_index(handler, mesh)
    apply_load!(f, NodalForce([uz], [1.0e4]), cache, asm, kernel, mesh)

    Kc = copy(K)
    apply_constraint!(Kc, bc)
    uc = Kc \ f
    @test all(isfinite, uc)
    ru = norm(Kc * uc - f) / max(norm(f), 1.0)
    @test ru < 1.0e-8
    @test maximum(abs.(uc[u_fix])) < 1.0e-4
    @test abs(uc[p_pin]) < 1.0e-4
end

@testset "StokesMixedKernel: matrix-free vs penalty-assembled K" begin
    mesh = brick_hex_mesh(2, 2, 2)
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}, p::DOF{Float64, Cell}}
    elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    kernel = StokesMixedKernel(ContinuumFormulation{FullThreeD}(); μ = 1.0e-3, inv_bulk = 0.0)
    asm = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elements, handler, mesh, kernel)
    assemble!(cache, asm, kernel, mesh)
    K, f = extract_system(cache)
    fill!(f, 0.0)

    u_fix = collect_vertex_dofs_on_nodeset(handler, mesh, :xmin)
    p_pin = default_pressure_gauge_dof(handler)
    fixed = Int[vcat(u_fix, [p_pin])...]
    bc = PenaltyDirichlet(fixed, zeros(Float64, length(fixed)); penalty = 1.0e14)

    uz = interior_uz_dof_index(handler, mesh)
    apply_load!(f, NodalForce([uz], [1.0e4]), cache, asm, kernel, mesh)

    Kc = copy(K)
    apply_constraint!(Kc, bc)

    mf = matrix_free_op(cache, asm, kernel, mesh; dirichlet = bc)
    @test LinearAlgebra.isposdef(mf) == false

    n = length(f)
    x = randn(n)
    y_asm = Kc * x
    y_mf = zeros(n)
    mul!(y_mf, mf, x)
    @test norm(y_mf - y_asm) < 1.0e-10 * max(norm(y_asm), 1.0)

    d = zeros(n)
    compute_diagonal!(d, cache, asm, kernel, mesh)
    apply_constraint_diag!(d, bc)
    @test maximum(abs.(d .- diag(Matrix(Kc)))) <
          1.0e-8 * max(maximum(abs.(diag(Matrix(Kc)))), 1.0)
end
