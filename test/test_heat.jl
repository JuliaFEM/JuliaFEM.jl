# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Test
using JuliaFEM.Preprocess

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

    # define boundary element for flux
    el2 = Element(Seg2, [1, 2])
    update!(el2, "geometry", X)
    # linear ramp from 0 -> 6 in time 0 -> 1
    update!(el2, "temperature flux", 0.0 => 0.0, 1.0 => 6.0)

    # define heat problem and push elements to problem
    problem = Problem(Heat, "one element heat problem", 1)
    problem.properties.formulation = "2D"
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
    solver = LinearSolver("solve heat problem")
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

function T_acc(x)
    # accurate solution
    a = 0.01
    L = 0.20
    k = 50.0
    Tᵤ = 20.0
    h = 10.0
    P = 4*a
    A = a^2
    α = h
    β = sqrt((h*P)/(k*A))
    T̂ = 100.0
    C = [1.0 1.0; (α+k*β)*exp(β*L) (α-k*β)*exp(-β*L)] \ [T̂-Tᵤ, 0.0]
    return dot(C, [exp(β*x), exp(-β*x)]) + Tᵤ
end

#=
@testset "test 1d heat problem" begin
    X = Dict{Int, Vector{Float64}}(
        1 => [0.0, 0.0, 0.0],
        2 => [0.1, 0.0, 0.0],
        3 => [0.2, 0.0, 0.0])
    e1 = Element(Seg2, [1, 2])
    e2 = Element(Seg2, [2, 3])
    e3 = Element(Poi1, [3])

    p1 = Problem(Heat, "1d heat problem", 1)
    p1.properties.formulation = "1D"
    push!(p1, e1, e2, e3)
    update!(p1, "geometry", X)
    a = 0.010
    update!(p1, "cross-section area", a^2)
    update!(p1, "cross-section perimeter", 4*a)
    update!(p1, "temperature thermal conductivity", 50.0) # k [W/(m∘C)]
    update!(p1, "temperature heat transfer coefficient", 10.0) # h [W/(m²∘C)]
    update!(p1, "temperature external temperature", 20.0)

    p2 = Problem(Dirichlet, "left boundary", 1, "temperature")
    e3 = Element(Poi1, [1])
    update!(e3, "geometry", X)
    update!(e3, "temperature 1", 100.0)
    push!(p2, e3)

    solver = LinearSolver(p1, p2)
    call(solver)
    T_min = minimum(p1.assembly.u)
    @test isapprox(T_max, T_acc(0.2); rtol=4.5e-2)
end
=#

@testset "test 3d heat problem" begin
    fn = Pkg.dir("JuliaFEM") * "/test/testdata/rod_short.med"
    mesh = aster_read_mesh(fn, "SHORT_ROD_RECTANGLE_HEX8")

    p1 = Problem(Heat, "rod", 1)
    push!(p1, create_elements(mesh, "ROD"))
    push!(p1, create_elements(mesh, "SIDES"))
    push!(p1, create_elements(mesh, "RIGHT"))
    update!(p1, "temperature thermal conductivity", 50.0)
    update!(p1, "temperature external temperature", 20.0)
    update!(p1, "temperature heat transfer coefficient", 10.0)

    p2 = Problem(Dirichlet, "left support T=100", 1, "temperature")
    push!(p2, create_elements(mesh, "LEFT"))
    update!(p2, "temperature 1", 100.0)

    solver = LinearSolver(p1, p2)
    call(solver)

    T_min = minimum(p1.assembly.u)

    # Code Aster solution
    T_CA_HEX20 = 4.58158267950429E+01
    T_CA_HEX8 = 3.77215189873436E+01
    info("T_min = $T_min")
    info("T_acc = $(T_acc(0.2))")
    rtol1 = norm(T_min-T_CA_HEX8)/max(T_min,T_CA_HEX8)*100.0
    rtol2 = norm(T_min-T_acc(0.2))/max(T_min,T_acc(0.2))*100.0
    info("rel. tol to CA solution: $rtol1 %")
    info("rel. tol to accurate solution: $rtol2 %")

    @test isapprox(T_min, T_acc(0.2); rtol=18.0e-2)
    @test isapprox(T_min, T_CA_HEX8; rtol=1.0e-9)
end
