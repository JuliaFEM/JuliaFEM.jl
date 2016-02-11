# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM.Test
using JuliaFEM.Core: Node, update!, Quad4, Seg2, Problem, Elasticity, Solver, Dirichlet

function get_test_problem()
    nodes = Dict{Int64, Node}(
        1 => [0.0, 0.0],
        2 => [1.0, 0.0],
        3 => [1.0, 1.0],
        4 => [0.0, 1.0])

    E = 288.0
    nu = 1.0/3.0
    f = -E/10.0
    expected = f/E*[-nu, 1]

    # field problem is plane stress linear elasticity
    element1 = Quad4([1, 2, 3, 4])
    element2 = Seg2([3, 4])
    update!([element1, element2], "geometry", nodes)
    update!(element1, "youngs modulus", E)
    update!(element1, "poissons ratio", nu)
#   update!(element2, "displacement traction force", [0.0, f])
    update!(element2, "displacement traction force", Vector{Float64}[[0.0, f], [0.0, f]])
    # type, name, dimension
    elasticity_problem = Problem(Elasticity, "block", 2)
    elasticity_problem.properties.formulation = :plane_stress
    push!(elasticity_problem, element1, element2)

    # dirichlet boundary condition, symmetry
    sym13 = Seg2([1, 2])
    sym23 = Seg2([4, 1])
    update!([sym13, sym23], "geometry", nodes)
    update!(sym13, "displacement 2", 0.0)
    update!(sym23, "displacement 1", 0.0)
    # type, name, dimension, unknown_field_name
    boundary_problem = Problem(Dirichlet, "symmetry boundaries", 2, "displacement")
    push!(boundary_problem, sym13, sym23)
    return elasticity_problem, boundary_problem
end

@testset "test 2d linear with surface load." begin

    E = 288.0
    nu = 1.0/3.0
    f = -E/10.0
    expected = f/E*[-nu, 1]
    elasticity_problem, boundary_problem = get_test_problem()
    solver = Solver("solve block problem")
    #solver.is_linear_system = true
    push!(solver, elasticity_problem)
    push!(solver, boundary_problem)
    call(solver)
    element1 = elasticity_problem.elements[1]
    u_disp = element1("displacement", [1.0, 1.0], 0.0)
    info("Displacement = $u_disp")
    @test isapprox(u_disp, expected)

end

