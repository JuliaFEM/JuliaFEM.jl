# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Test


@testset "test one element heat problem" begin

    X = Dict{Int, Vector{Float64}}(
        1 => [0.0,0.0],
        2 => [1.0,0.0],
        3 => [1.0,1.0],
        4 => [0.0,1.0])

    # define volume element
    el1 = Element(Quad4, [1, 2, 3, 4])

    update!(el1, "geometry", X)
    update!(el1, "temperature thermal conductivity", 6.0)
    update!(el1, "temperature load", 12.0)
    update!(el1, "density", 36.0)

    # define boundary element for flux
    el2 = Element(Seg2, [1, 2])
    update!(el2, "geometry", X)
    # linear ramp from 0 -> 6 in time 0 -> 1
    update!(el2, "temperature flux", 0.0 => 0.0, 1.0 => 6.0)

    # define heat problem and push elements to problem
    problem = Problem(Heat, "one element heat problem", 1)
    push!(problem, el1, el2)

    # define boundary element for dirichlet boundary condition
    el3 = Element(Seg2, [3, 4])
    update!(el3, "geometry", X)
    update!(el3, "temperature 1", 0.0)

    boundary_condition = Problem(Dirichlet, "T=0 on top", 1, "temperature")
    push!(boundary_condition, el3)

    # manual assembling of problem + solution:
    assemble!(problem, 0.0)
    A = full(problem.assembly.K)
    b = full(problem.assembly.f)
    A_expected = [
         4.0 -1.0 -2.0 -1.0
        -1.0  4.0 -1.0 -2.0
        -2.0 -1.0  4.0 -1.0
        -1.0 -2.0 -1.0  4.0]
    free_dofs = [1, 2]
    @test isapprox(A, A_expected)
    @test isapprox(A[free_dofs, free_dofs] \ b[free_dofs], [1.0, 1.0])

    # using Solver
    solver = Solver("solve heat problem")
    solver.is_linear_system = true
    push!(solver, problem, boundary_condition)

    # Set constant source f=12 with k=6. Accurate solution is
    # T=1 on free boundary, u(x,y) = -1/6*(1/2*f*x^2 - f*x)
    # when boundary flux not active (at t=0)
    solver.time = 0.0
    call(solver)
    # interpolate temperature at middle of element 2 (flux boundary) at time t=0:
    T = el2("temperature", [0.0], 0.0)
    @test isapprox(T[1], 1.0)

    # Set constant flux g=6 on boundary. Accurate solution is
    # u(x,y) = x which equals T=1 on boundary.
    # at time t=1.0 all loads should be on.
    solver.time = 1.0
    call(solver)
    T = el2("temperature", [0.0], 1.0)
    @test isapprox(T[1], 2.0)
end

