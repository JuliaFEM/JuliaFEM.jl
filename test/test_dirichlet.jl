# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM.Test
using JuliaFEM.Core: Tri3, Seg2, Dirichlet, Assembly, assemble!, Node, Problem

#=
@testset "dirichlet problem in 1 dimension" begin
    element = Seg2([1, 2])
    element["geometry"] = Node[[1.0, 1.0], [0.0, 1.0]]
    element["temperature"] = 0.0
    problem = Problem(Dirichlet, "test problem", 1, "temperature")
    push!(problem, element)
    assemble!(problem, 0.0)
    C1 = full(problem.assembly.C1)
    info("C1")
    dump(C1)
    C2 = full(problem.assembly.C2)
    g = full(problem.assembly.g)
    @test isapprox(C1, C2)
    @test isapprox(C1, 1/6*[2 1; 1 2])
    @test isapprox(g, [0.0, 0.0])
end

@testset "dirichlet problem using tri3 surface element" begin
    element = Tri3([1, 2, 3])
    element["geometry"] = Node[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    element["temperature"] = 0.0
    problem = Problem(Dirichlet, "test problem", 1, "temperature")
    push!(problem, element)
    assemble!(problem, 0.0)
    C1 = full(problem.assembly.C1)
    C2 = full(problem.assembly.C2)
    @test isapprox(C1, C2)
    @test isapprox(C1, 1/24*[2 1 1; 1 2 1; 1 1 2])
end
=#

@testset "dirichlet problem in 2 dimensions" begin
    element = Seg2([1, 2])
    element["geometry"] = Node[[1.0, 1.0], [0.0, 1.0]]
    element["displacement 1"] = 0.0
    element["displacement 2"] = 0.0
    problem = Problem(Dirichlet, "test problem", 2, "displacement")
    push!(problem, element)
    assemble!(problem, 0.0)
    C1 = full(problem.assembly.C1)
    C2 = full(problem.assembly.C2)
    g = full(problem.assembly.g)
    @test isapprox(C1, C2)
    C1_expected = 1/6*[2 0 1 0; 0 2 0 1; 1 0 2 0; 0 1 0 2]
    @test isapprox(C1, C1_expected)
    @test isapprox(g, [0.0, 0.0, 0.0, 0.0])
end

@testset "dirichlet problem in 2 dimensions, with 1 dof fixed" begin
    element = Seg2([1, 2])
    element["geometry"] = Node[[1.0, 1.0], [0.0, 1.0]]
    element["displacement 2"] = 0.0
    problem = Problem(Dirichlet, "test problem", 2, "displacement")
    push!(problem, element)
    assemble!(problem, 0.0)
    C1 = full(problem.assembly.C1)
    C2 = full(problem.assembly.C2)
    g = full(problem.assembly.g)
    @test isapprox(C1, C2)
    C1_expected = 1/6*[
        0 0 0 0
        0 2 0 1
        0 0 0 0
        0 1 0 2]
    @test isapprox(C1, C1_expected)
    @test isapprox(g, [0.0, 0.0, 0.0, 0.0])
end

