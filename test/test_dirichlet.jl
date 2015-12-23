# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

module TestDirichletBoundaryCondition

using JuliaFEM.Test
using JuliaFEM.Core: Tri3, Seg2, DirichletProblem, Assembly, assemble, Node,
                     BiorthogonalBasis

@testset "test dirichlet boundary conditions" begin

@testset "dirichlet problem in 1 dimension" begin
    element = Seg2([1, 2])
    element["geometry"] = Node[[1.0, 1.0], [0.0, 1.0]]
    element["temperature"] = 0.0
    problem = DirichletProblem("temperature", 1)
    push!(problem, element)
    assembly = assemble(problem, 0.0)
    C1 = full(assembly.C1)
    C2 = full(assembly.C2)
    g = full(assembly.g)
    @test isapprox(C1, C2)
    @test isapprox(C1, 1/6*[2 1; 1 2])
    @test isapprox(g, [0.0, 0.0])
end

@testset "dirichlet problem in 2 dimensions" begin
    element = Seg2([1, 2])
    element["geometry"] = Node[[1.0, 1.0], [0.0, 1.0]]
    element["displacement"] = 0.0
    problem = DirichletProblem("displacement", 2)
    push!(problem, element)
    assembly = assemble(problem, 0.0)
    C1 = full(assembly.C1)
    C2 = full(assembly.C2)
    g = full(assembly.g)
    @test isapprox(C1, C2)
    C1_expected = 1/6*[2 0 1 0; 0 2 0 1; 1 0 2 0; 0 1 0 2]
    @test isapprox(C1, C1_expected)
    @test isapprox(g, [0.0, 0.0, 0.0, 0.0])
end

@testset "dirichlet problem in 2 dimensions, with 1 dof fixed" begin
    element = Seg2([1, 2])
    element["geometry"] = Node[[1.0, 1.0], [0.0, 1.0]]
    element["displacement 2"] = 0.0
    problem = DirichletProblem("displacement", 2)
    push!(problem, element)
    assembly = assemble(problem, 0.0)
    C1 = full(assembly.C1)
    C2 = full(assembly.C2)
    g = full(assembly.g)
    @test isapprox(C1, C2)
    C1_expected = 1/6*[
        0 0 0 0
        0 2 0 1
        0 0 0 0
        0 1 0 2]
    @test isapprox(C1, C1_expected)
    @test isapprox(g, [0.0, 0.0, 0.0, 0.0])
end

@testset "dirichlet problem using tri3 surface element" begin
    elem = Tri3([1, 2, 3])
    elem["geometry"] = Node[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    elem["temperature"] = 0.0
    prob = DirichletProblem("temperature", 1)
    push!(prob, elem)
    ass = assemble(prob, 0.0)
    C1 = full(ass.C1)
    C2 = full(ass.C2)
    @test isapprox(C1, C2)
    @test isapprox(C1, 1/24*[2 1 1; 1 2 1; 1 1 2])
end

@testset "dirichlet problem using biorthogonal basis" begin
    elem = Tri3([1, 2, 3])
    elem["geometry"] = Node[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    elem["displacement 1"] = 1.0
    prob = DirichletProblem("displacement", 3; basis=BiorthogonalBasis)
    push!(prob, elem)
    ass = assemble(prob, 0.0)
    C1 = full(ass.C1, 9, 9)
    C2 = full(ass.C2, 9, 9)
    D = full(ass.D, 9, 9)
    g = full(ass.g, 9, 1)
    C1_expected = eye(9)*1.0/6.0
    g_expected = zeros(9)
    g_expected[1] = g_expected[4] = g_expected[7] = 1.0/6.0
    C2_expected = zeros(9, 9)
    D_expected = zeros(9, 9)
    C2_expected[1, 1] = 1.0/6.0
    D_expected[2, 2] = 1.0/6.0
    D_expected[3, 3] = 1.0/6.0
    C2_expected[4, 4] = 1.0/6.0
    D_expected[5, 5] = 1.0/6.0
    D_expected[6, 6] = 1.0/6.0
    C2_expected[7, 7] = 1.0/6.0
    D_expected[8, 8] = 1.0/6.0
    D_expected[9, 9] = 1.0/6.0
    @test isapprox(C1, C1_expected)
    @test isapprox(C2, C2_expected)
    @test isapprox(D, D_expected)
    @test isapprox(g, g_expected)
end

end

end
