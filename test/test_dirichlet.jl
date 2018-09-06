# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM, SparseArrays, Test

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

    time = 0.0

    element = Element(Seg2, (1, 2))
    X = Dict(1 => [0.0, 0.0], 2 => [6.0, 0.0])
    update!(element, "geometry", X)
    update!(element, "temperature 1", 0.0)

    problem1 = Problem(Dirichlet, "test problem 1", 1, "temperature")
    problem1.properties.variational = true
    problem1.properties.dual_basis = false
    add_element!(problem1, element)
    assemble!(problem1, time)
    C1 = problem1.assembly.C1
    C2 = problem1.assembly.C2
    @test isapprox(C1, C2)
    @test isapprox(C1, [2.0 1.0; 1.0 2.0])

    problem2 = Problem(Dirichlet, "test problem 2", 1, "temperature")
    problem2.properties.variational = true
    problem2.properties.dual_basis = true
    add_element!(problem2, element)
    assemble!(problem2, time)
    C1 = problem2.assembly.C1
    C2 = problem2.assembly.C2
    @test isapprox(C1, C2)
    @test isapprox(C1, [3.0 0.0; 0.0 3.0])

    element = Element(Seg3, (1, 2, 3))
    X = Dict(1 => [0.0, 0.0], 2 => [30.0, 0.0], 3 => [15.0, 0.0])
    update!(element, "geometry", X)
    update!(element, "temperature 1", 0.0)
    problem3 = Problem(Dirichlet, "quadratic 1", 1, "temperature")
    problem3.properties.variational = true
    problem3.properties.dual_basis = false
    add_element!(problem3, element)
    assemble!(problem3, time)
    C1 = problem3.assembly.C1
    C2 = problem3.assembly.C2
    @test isapprox(C1, C2)
    @test isapprox(C1, [4.0 -1.0 2.0; -1.0 4.0 2.0; 2.0 2.0 16.0])

    problem4 = Problem(Dirichlet, "quadratic 2", 1, "temperature")
    problem4.properties.variational = true
    problem4.properties.dual_basis = true
    add_element!(problem4, element)
    assemble!(problem4, time)
    C1 = problem4.assembly.C1
    C2 = problem4.assembly.C2
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
    X = Dict(1 => [0.0, 0.0],
             2 => [1.0, 0.0])
    element = Element(Seg2, (1, 2))
    update!(element, "geometry", X)
    function f(element, ip, time)
        x, y = element("geometry", ip, time)
        val = x*time
        @debug("analytical function called", ip, time, x, y, val)
        return val
    end
    update!(element, "displacement 1", 0.0)
    update!(element, "displacement 2", f)
    problem = Problem(Dirichlet, "test boundary", 2, "displacement")
    add_element!(problem, element)
    time = 0.0
    assemble!(problem, time)
    @test isapprox(problem.assembly.g, [0.0, 0.0, 0.0, 0.0])
    empty!(problem.assembly)
    time = 1.0
    assemble!(problem, time)
    g2 = Vector(problem.assembly.g, 4)
    C2 = Matrix(problem.assembly.C2, 4, 4)
    u = C2 \ g2
    @debug("displacement vector", u)
    @test isapprox(u, [0.0, 0.0, 0.0, 1.0])
end
