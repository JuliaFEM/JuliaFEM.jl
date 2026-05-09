# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Small `Hex8` brick: `ContinuumKernel` + `PenaltyDirichlet` on `xmin` + nodal
`z` load, direct sparse solve, relative residual check.
"""

using Test
using JuliaFEM
using JuliaFEM: ContinuumKernel, ContinuumFormulation, FullThreeD, LinearElastic
using JuliaFEM: DOFBasedCOOAssembler, DOFBasedCOOCache, assemble!, extract_system
using JuliaFEM: PenaltyDirichlet, apply_constraint!, apply_load!, NodalForce
using JuliaFEM: @DOFSet, DOF, Displacement, Vertex, Hex8
using LinearAlgebra

include("solve_brick_helpers.jl")

@testset "ContinuumKernel: brick direct solve (xmin fixed + uz load)" begin
    mesh = brick_hex_mesh(2, 2, 2)
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    kernel = ContinuumKernel(ContinuumFormulation{FullThreeD}(), LinearElastic(E = 210e9, ν = 0.3))
    asm = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elements, handler, mesh, kernel)
    assemble!(cache, asm, kernel, mesh)
    K, f = extract_system(cache)
    fill!(f, 0.0)

    u_fix = collect_vertex_dofs_on_nodeset(handler, mesh, :xmin)
    uz = interior_uz_dof_index(handler, mesh)
    λ = 1.0e14
    bc = PenaltyDirichlet(u_fix, zeros(Float64, length(u_fix)); penalty = λ)
    apply_load!(f, NodalForce([uz], [1.0e4]), cache, asm, kernel, mesh)

    Kc = copy(K)
    apply_constraint!(Kc, bc)
    uc = Kc \ f
    @test all(isfinite, uc)
    ru = norm(Kc * uc - f) / max(norm(f), 1.0)
    @test ru < 1.0e-8
    @test maximum(abs.(uc[u_fix])) < 1.0e-4
end
