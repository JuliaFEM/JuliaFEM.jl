# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

module SolverTests

using JuliaFEM.Test

using JuliaFEM.Core: Seg2, Quad4
using JuliaFEM.Core: DirichletProblem, HeatProblem
using JuliaFEM.Core: LinearSolver

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
    solver = LinearSolver(field_problem, boundary_problem)

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

test_basic()

end
