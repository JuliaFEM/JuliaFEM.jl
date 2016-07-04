# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Test

#=
In [36]: C = Matrix([[0], [30], [15]]) # node coordinates
In [37]: A = Matrix([P.subs({x: C[i,0]}).T for i in range(len(P))])
In [38]: N = P.T*A.inv()
In [39]: Me = integrate(N.T*N, (x, 0, 30))
In [40]: De = diag(*integrate(N, (x, 0, 30)))
In [41]: Me
Out[41]: 
Matrix([
[ 4, -1,  2],
[-1,  4,  2],
[ 2,  2, 16]])
In [42]: De
Out[42]: 
Matrix([
[5, 0,  0],
[0, 5,  0],
[0, 0, 20]])
=#

@testset "dirichlet problem in 1 dimension" begin
    element = Element(Seg2, [1, 2])
    element["geometry"] = Vector{Float64}[[0.0, 0.0], [6.0, 0.0]]
    element["temperature 1"] = 0.0
    p1 = Problem(Dirichlet, "test problem 1", 1, "temperature")
    p1.properties.dual_basis = false
    p2 = Problem(Dirichlet, "test problem 2", 1, "temperature")
    p2.properties.dual_basis = true
    assemble!(p1, element)
    assemble!(p2, element)
    C1 = full(p1.assembly.C1)
    C2 = full(p1.assembly.C2)
    @test isapprox(C1, C2)
    @test isapprox(C1, [2.0 1.0; 1.0 2.0])
    C1 = full(p2.assembly.C1)
    C2 = full(p2.assembly.C2)
    @test isapprox(C1, C2)
    @test isapprox(C1, [3.0 0.0; 0.0 3.0])

    element = Element(Seg3, [1, 2, 3])
    element["geometry"] = Vector{Float64}[[0.0, 0.0], [30.0, 0.0], [15.0, 0.0]]
    element["temperature 1"] = 0.0
    p1 = Problem(Dirichlet, "quadratic 1", 1, "temperature")
    p1.properties.dual_basis = false
    p2 = Problem(Dirichlet, "quadratic 1", 1, "temperature")
    p2.properties.dual_basis = true
    assemble!(p1, element)
    assemble!(p2, element)
    C1 = full(p1.assembly.C1)
    C2 = full(p1.assembly.C2)
    @test isapprox(C1, C2)
    @test isapprox(C1, [4.0 -1.0 2.0; -1.0 4.0 2.0; 2.0 2.0 16.0])
    C1 = full(p2.assembly.C1)
    C2 = full(p2.assembly.C2)
    @test isapprox(C1, C2)
    @test isapprox(C1, [5.0 0.0 0.0; 0.0 5.0 0.0; 0.0 0.0 20.0])
end

#=
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
=#

@testset "test analytical boundary condition" begin
    X = Dict{Int64, Vector{Float64}}(
        1 => [0.0, 0.0],
        2 => [1.0, 0.0])
    element = Element(Seg2, [1, 2])
    update!(element, "geometry", X)
    update!(element, "displacement 1", 0.0)
    f(xi, time) = begin
        info("function called at xi = $xi, time = $time")
        X = element("geometry", xi, time)
        info("geometry at xi, X = $X")
        val = X[1]*time
        info("result for field at xi = $val")
        return val
    end
    update!(element, "displacement 2", f)
    p = Problem(Dirichlet, "test boundary", 2, "displacement")
    push!(p, element)
    assemble!(p, 0.0)
    g1 = full(p.assembly.g, 4, 1)
    @test isapprox(g1, [0.0, 0.0, 0.0, 0.0])
    empty!(p.assembly)
    assemble!(p, 1.0)
    g2 = full(p.assembly.g, 4, 1)
    C2 = full(p.assembly.C2, 4, 4)
    u = C2 \ g2
    info("u = $u")
    @test isapprox(u, [0.0, 0.0, 0.0, 1.0])
end

