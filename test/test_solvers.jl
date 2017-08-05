# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Testing

@testset "test linearsolver + xdmf writing" begin
    el1 = Element(Quad4, [1, 2, 3, 4])
    el2 = Element(Seg2, [1, 2])
    el3 = Element(Seg2, [3, 4])
    X = Dict(
        1 => [0.0, 0.0],
        2 => [1.0, 0.0],
        3 => [1.0, 1.0],
        4 => [0.0, 1.0])
    update!([el1, el2, el3], "geometry", X)
    update!(el1, "thermal conductivity", 6.0)
    update!(el1, "density", 36.0)

    update!(el2, "heat flux", 0.0 => 0.0)
    update!(el2, "heat flux", 1.0 => 600.0)

    problem = Problem(Heat, "test problem", 1)
    problem.properties.formulation = "2D"
    push!(problem.elements, el1, el2)

    update!(el3, "temperature 1", 0.0)

    bc = Problem(Dirichlet, "fixed", 1, "temperature")

    push!(bc.elements, el3)

    # Create a solver for a set of problems
    solver = Solver(Linear, "solve heat problem")
    push!(solver, problem, bc)

    # Solve problem at time t=1.0 and update fields
    solver.time = 1.0
    solver.xdmf = Xdmf()
    solver()

    # Postprocess.
    # Interpolate temperature field along boundary of Γ₁ at time t=1.0
    xi = (0.0, )
    X = el2("geometry", xi, 1.0)
    T = el2("temperature", xi, 1.0)
    info("Temperature at point X = $X is T = $T")
    @test isapprox(T, 100.0)
end

@testset "problem not found from solver" begin
    s = Solver(Linear, "demo solver")
    @test_throws KeyError getindex(s, "not_found")
end

@testset "automatic determination of problem dimension if not spesified" begin
    s = Solver(Linear, "demo solver")
    p = Problem(Elasticity, "demo problem", 2)
    push!(s, p)
    get_field_assembly(s)
    @test s.ndofs == 0
    add!(p.assembly.K, [4], [4], reshape([4.0],1,1))
    get_field_assembly(s)
    @test s.ndofs == 4
end

@testset "test for error when overdetermined system and requesting boundary assembly" begin
    s = Solver(Linear, "demo solver")
    @test_throws AssertionError get_boundary_assembly(s) # ndofs = 0
    p1 = Problem(Dirichlet, "bc1", 2, "displacement")
    p2 = Problem(Dirichlet, "bc2", 2, "displacement")
    # third dofs constrained 
    add!(p1.assembly.C2, [3], [3], reshape([1.0],1,1))
    add!(p2.assembly.C2, [3], [4], reshape([1.0],1,1))
    s.ndofs = 4
    push!(s, p1, p2)
    @test_throws ErrorException get_boundary_assembly(s)
end
