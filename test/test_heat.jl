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

    # volume element
    element = Element(Quad4, [1, 2, 3, 4])

    update!(element, "geometry", X)
    update!(element, "temperature thermal conductivity", 6.0)
    update!(element, "temperature load", [12.0, 12.0, 12.0, 12.0])
    update!(element, "density", 36.0)

    # boundary element 
    boundary_element = Element(Seg2, [1, 2])
    update!(boundary_element, "geometry", X)
    # linear ramp from 0 to 6 in time 0 to 1
    update!(boundary_element, "temperature flux", 0.0 => 0.0, 1.0 => 6.0)

    problem = Problem(Heat, "one element heat problem", 1)
    push!(problem, element, boundary_element)

    # Set constant source f=12 with k=6. Accurate solution is
    # T=1 on free boundary, u(x,y) = -1/6*(1/2*f*x^2 - f*x)
    assemble!(problem, 0.0)
    A = full(problem.assembly.K)
    b = full(problem.assembly.f)

    A_expected = [
         4.0 -1.0 -2.0 -1.0
        -1.0  4.0 -1.0 -2.0
        -2.0 -1.0  4.0 -1.0
        -1.0 -2.0 -1.0  4.0]

    @test isapprox(A, A_expected)

    free_dofs = [1, 2]
    @test isapprox(A[free_dofs, free_dofs] \ b[free_dofs], [1.0, 1.0])

    # Set constant flux g=6 on boundary. Accurate solution is
    # u(x,y) = x which equals T=1 on boundary.
    # at time t=1.0 all loads should be on.
    empty!(problem.assembly)
    assemble!(problem, 1.0)
    A = full(problem.assembly.K)
    b = full(problem.assembly.f)
    T = A[free_dofs, free_dofs] \ b[free_dofs]
    @test isapprox(T, [2.0, 2.0])
end

