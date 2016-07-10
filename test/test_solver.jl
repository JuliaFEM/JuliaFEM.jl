# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Testing

function test_linearsolver()
    el1 = Quad4([1, 2, 3, 4])
    el1["geometry"] = Vector[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    el1["temperature thermal conductivity"] = 6.0
    el1["density"] = 36.0

    el2 = Seg2([1, 2])
    el2["geometry"] = Vector[[0.0, 0.0], [1.0, 0.0]]
    el2["temperature flux"] = (
        (0.0 => 0.0),
        (1.0 => 600.0)
        )

    field_problem = HeatProblem()
    push!(field_problem, el1)
    push!(field_problem, el2)

    el3 = Seg2([3, 4])
    el3["geometry"] = Vector[[1.0, 1.0], [0.0, 1.0]]
    el3["temperature"] = 0.0

    boundary_problem = DirichletProblem("temperature", 1)

    push!(boundary_problem, el3)

    # Create a solver for a set of problems
    solver = LinearSolver()
    push!(solver, field_problem)
    push!(solver, boundary_problem)

    # Solve problem at time t=1.0 and update fields
    solver(1.0)

    # Postprocess.
    # Interpolate temperature field along boundary of Γ₁ at time t=1.0
    xi = [0.0, -1.0]
    X = el2("geometry", xi, 1.0)
    T = el2("temperature", xi, 1.0)
    info("Temperature at point X = $X is T = $T")
    @test isapprox(T, 100.0)
end

function test_solvers()
    K =  [
    440.0   150.0  -260.0   -30.0    40.0    30.0  -220.0  -150.0
    150.0   440.0    30.0    40.0   -30.0  -260.0  -150.0  -220.0
   -260.0    30.0   440.0  -150.0  -220.0   150.0    40.0   -30.0
    -30.0    40.0  -150.0   440.0   150.0  -220.0    30.0  -260.0
     40.0   -30.0  -220.0   150.0   440.0  -150.0  -260.0    30.0
     30.0  -260.0   150.0  -220.0  -150.0   440.0   -30.0    40.0
   -220.0  -150.0    40.0    30.0  -260.0   -30.0   440.0   150.0
   -150.0  -220.0   -30.0  -260.0    30.0    40.0   150.0   440.0]

    C = 1/3*[
    1.0  0.0  0.0  0.0  0.5  0.0  0.0  0.0
    0.0  1.0  0.0  0.0  0.0  0.5  0.0  0.0
    0.0  0.0  1.0  0.0  0.0  0.0  0.5  0.0
    0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.5
    0.5  0.0  0.0  0.0  1.0  0.0  0.0  0.0
    0.0  0.5  0.0  0.0  0.0  1.0  0.0  0.0
    0.0  0.0  0.5  0.0  0.0  0.0  1.0  0.0
    0.0  0.0  0.0  0.5  0.0  0.0  0.0  1.0]

    f = zeros(8)

    g = 1/100 * [-5.0, 5.0, 10.0, -10.0, -5.0, 5.0, 10.0, -10.0]

    K = sparse(K)
    C = sparse(C)
    f = sparse(f)
    g = sparse(g)

    expected = [-0.1, 0.1, 0.2, -0.2, -0.1, 0.1, 0.2, -0.2]
    u1, la1 = solve(K, f, C, g, Val{:UMFPACK})
    @test isapprox(u1, expected)
    u2, la2 = solve(K, f, C, g, Val{:CHOLMOD})
    @test isapprox(u2, expected)
    # FIXME: how to dynamically include packages only if they are installed?
    #include(Pkg.dir("JuliaFEM"*"/src/petsc.jl"))
    u3, la3 = solve(K, f, C, g, Val{:PETSc_GMRES})
    @test isapprox(u3, expected)
end

