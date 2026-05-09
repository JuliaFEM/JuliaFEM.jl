# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
`HellingerReissnerKernel`: penalty Dirichlet on `xmin` displacement + one
stress gauge DOF; assembled `K` vs `MatrixFreeOperator` matvec on random `x`.
"""

using Test
using JuliaFEM
using JuliaFEM: HellingerReissnerKernel, ContinuumFormulation, FullThreeD, LinearElastic
using JuliaFEM: DOFBasedCOOAssembler, DOFBasedCOOCache, assemble!, extract_system
using JuliaFEM: PenaltyDirichlet, apply_constraint!, apply_load!
using JuliaFEM: compute_diagonal!, apply_constraint_diag!
using JuliaFEM: matrix_free_op
using JuliaFEM: @DOFSet, DOF, Displacement, Vertex, Hex8
using LinearAlgebra
using SparseArrays

include("solve_brick_helpers.jl")

@testset "HellingerReissnerKernel: matrix-free vs penalty-assembled K" begin
    mesh = brick_hex_mesh(2, 2, 2)
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}, sig::DOF{SymmetricTensor{2,3}, Cell}}
    elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    kernel = HellingerReissnerKernel(ContinuumFormulation{FullThreeD}(), LinearElastic(E = 210e9, ν = 0.3))
    asm = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elements, handler, mesh, kernel)
    assemble!(cache, asm, kernel, mesh)
    K, f = extract_system(cache)
    fill!(f, 0.0)

    u_fix = collect_vertex_dofs_on_nodeset(handler, mesh, :xmin)
    σ_pin = first_field_dof(handler, 2, 1)
    fixed = Int[vcat(u_fix, [σ_pin])...]
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
