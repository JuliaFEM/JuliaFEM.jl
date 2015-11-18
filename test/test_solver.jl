# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

module SolverTests

using JuliaFEM.Test
using JuliaFEM

using JuliaFEM: DirichletProblem, Seg2, PlaneHeatProblem, Quad4, SimpleSolver, get_element, get_basis

""" Define Problem 1:

- Field function: Laplace equation Δu=0 in Ω={u∈R²|(x,y)∈[0,1]×[0,1]}
- Neumann boundary on Γ₁={0<=x<=1, y=0}, ∂u/∂n=600 on Γ₁
"""
function get_heatproblem()
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

    problem1 = PlaneHeatProblem()
    push!(problem1, el1)
    push!(problem1, el2)
    return problem1
end

""" Define Problem 2:
 - Dirichlet boundary Γ₂={0<=x<=1, y=1}, u=0 on Γ₂
"""
function get_boundaryproblem()
    el3 = Seg2([3, 4])
    el3["geometry"] = Vector[[1.0, 1.0], [0.0, 1.0]]
    problem2 = DirichletProblem(1)
    push!(problem2, el3)
    return problem2
end

function test_simplesolver()
    info("construct heat problem")
    problem1 = get_heatproblem()
    info("construct boundary problem")
    problem2 = get_boundaryproblem()
    # Create a solver for a set of problems
    info("create SimpleSolver with problems.")
    solver = SimpleSolver()
    push!(solver, problem1)
    push!(solver, problem2)
    info("solve!")
    # Solve problem at time t=1.0 and update fields
    call(solver, 1.0)
    # Postprocess.
    # Interpolate temperature field along boundary of Γ₁ at time t=1.0
    xi = [0.0, -1.0]
    el2 = get_element(problem1.equations[2])
    basis = get_basis(el2)
    X = basis("geometry", xi, 1.0)
    T = basis("temperature", xi, 1.0)
    info("Temperature at point X = $X is T = $T")
    @test isapprox(T, 100.0)
end

end
