# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using Base.Test

@testset "two increments, linear solver" begin
    X = Dict{Int, Vector{Float64}}(
        1 => [0.0,0.0],
        2 => [1.0,0.0],
        3 => [1.0,1.0],
        4 => [0.0,1.0])
    element = Element(Quad4, [1, 2, 3, 4])
    update!(element, "geometry", X)
    update!(element, "thermal conductivity", 6.0)
    update!(element, "heat source", 0.0 => 12.0)
    update!(element, "heat source", 1.0 => 24.0)
    problem = Problem(PlaneHeat, "one element heat problem", 1)
    push!(problem, element)
    boundary_element = Element(Seg2, [1, 2])
    update!(boundary_element, "geometry", X)
    update!(boundary_element, "temperature 1", 0.0)
    bc = Problem(Dirichlet, "fixed", 1, "temperature")
    push!(bc, boundary_element)
    solver = Solver(Linear, problem, bc)

    empty!(problem.assembly)
    solve!(solver, 0.0)
    @test isapprox(solver("temperature", 0.0)[3], 1.0)

    empty!(problem.assembly)
    solver()
    @test isapprox(solver("temperature", 0.0)[3], 1.0)

    empty!(problem.assembly)
    solve!(solver, 1.0)
    @test isapprox(solver("temperature", 1.0)[3], 2.0)

    empty!(problem.assembly)
    solver()
    @test isapprox(solver("temperature", 1.0)[3], 2.0)

end

@testset "two increments, nonlinear solver" begin
    X = Dict{Int, Vector{Float64}}(
        1 => [0.0,0.0],
        2 => [1.0,0.0],
        3 => [1.0,1.0],
        4 => [0.0,1.0])
    element = Element(Quad4, [1, 2, 3, 4])
    update!(element, "geometry", X)
    update!(element, "thermal conductivity", 6.0)
    update!(element, "heat source", 0.0 => 12.0)
    update!(element, "heat source", 1.0 => 24.0)
    problem = Problem(PlaneHeat, "one element heat problem", 1)
    push!(problem, element)
    boundary_element = Element(Seg2, [1, 2])
    update!(boundary_element, "geometry", X)
    update!(boundary_element, "temperature 1", 0.0)
    bc = Problem(Dirichlet, "fixed", 1, "temperature")
    push!(bc, boundary_element)
    solver = Solver(Nonlinear, problem, bc)

    empty!(problem.assembly)
    solve!(solver, 0.0)
    @test isapprox(solver("temperature", 0.0)[3], 1.0)

    empty!(problem.assembly)
    solver()
    @test isapprox(solver("temperature", 0.0)[3], 1.0)

    empty!(problem.assembly)
    solve!(solver, 1.0)
    @test isapprox(solver("temperature", 1.0)[3], 2.0)

    empty!(problem.assembly)
    solve!(solver, 1.0)
    @test isapprox(solver("temperature", 1.0)[3], 2.0)

end
