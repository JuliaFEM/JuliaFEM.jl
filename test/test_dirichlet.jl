# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

module TestDirichletBoundaryCondition

using JuliaFEM.Test
using JuliaFEM.Core: Tri3, Seg2, DirichletProblem, Assembly, assemble

function test_dirichlet_problem_1_dim()
    element = Seg2([1, 2])
    element["geometry"] = Vector[[1.0, 1.0], [0.0, 1.0]]
    element["temperature"] = 0.0
    problem = DirichletProblem("temperature", 1)
    push!(problem, element)
    assembly = assemble(problem, 0.0)
    A = full(assembly.stiffness_matrix)
    b = full(assembly.force_vector)
    @test isapprox(A, 1/6*[2 1; 1 2])
    @test isapprox(b, [0.0, 0.0])
end

function test_dirichlet_problem_2_dim()
    element = Seg2([1, 2])
    element["geometry"] = Vector[[1.0, 1.0], [0.0, 1.0]]
    element["displacement"] = 0.0
    problem = DirichletProblem("displacement", 2)
    push!(problem, element)
    assembly = assemble(problem, 0.0)
    A = full(assembly.stiffness_matrix)
    b = full(assembly.force_vector)
    A_expected = 1/6*[2 0 1 0; 0 2 0 1; 1 0 2 0; 0 1 0 2]
    @test isapprox(A, A_expected)
    @test isapprox(b, [0.0, 0.0, 0.0, 0.0])
end

function test_dirichlet_problem_2_dim_single_dof_fixed()
    element = Seg2([1, 2])
    element["geometry"] = Vector[[1.0, 1.0], [0.0, 1.0]]
    element["displacement 2"] = 0.0
    problem = DirichletProblem("displacement", 2)
    push!(problem, element)
    assembly = assemble(problem, 0.0)
    A = full(assembly.stiffness_matrix)
    b = full(assembly.force_vector)
    info(b)
    info("A = \n$A")
    A_expected = 1/6*[
        0 0 0 0
        0 2 0 1
        0 0 0 0
        0 1 0 2]
    @test isapprox(A, A_expected)
    @test isapprox(b, [0.0, 0.0, 0.0, 0.0])
end

function test_dirichlet_surface_tri3()
    elem = Tri3([1, 2, 3])
    elem["geometry"] = Vector{Float64}[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    elem["temperature"] = 0.0
    prob = DirichletProblem("temperature", 1)
    push!(prob, elem)
    ass = assemble(prob, 0.0)
    k = full(ass.stiffness_matrix)
    @test isapprox(k, 1/24*[2 1 1; 1 2 1; 1 1 2])
end

end
