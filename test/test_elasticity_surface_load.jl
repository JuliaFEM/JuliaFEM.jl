# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM.Test
using JuliaFEM.Core: Node, update!, Quad4, Seg2, PlaneStressLinearElasticityProblem,
                     assemble, FieldProblem, BoundaryProblem, DirichletProblem,
                     DirectSolver, DualBasis

@testset "test 2d linear elasticity with surface load." begin

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
    elasticity_problem = FieldProblem(PlaneStressLinearElasticityProblem, "block", 2)
    push!(elasticity_problem, element1, element2)

    # dirichlet boundary condition, symmetry
    sym13 = Seg2([1, 2])
    sym23 = Seg2([4, 1])
    update!([sym13, sym23], "geometry", nodes)
    update!(sym13, "displacement 2", 0.0)
    update!(sym23, "displacement 1", 0.0)
    #sym23["displacement 1"] = 0.0
    #sym13["displacement 2"] = 0.0
    # name, unknown field, unknown field dimension
    boundary_problem = BoundaryProblem(DirichletProblem, "symmetry boundaries", "displacement", 2)
    push!(boundary_problem, sym13, sym23)

    solver = DirectSolver("solve block problem")
    solver.solve_residual = false
    push!(solver, elasticity_problem)
    push!(solver, boundary_problem)

    #=
    add_linear_system_solver_posthook!(solver,
    function (solver)
        info("solution vector")
        dump(solver.x)
    end)
    =#

    call(solver, 0.0)

    #=
    free_dofs = Int64[3, 5, 6, 8]
    ass = assemble(problem, 0.0)
    f = full(ass.force_vector)
    K = full(ass.stiffness_matrix)
    u = zeros(2, 4)
    u[free_dofs] = K[free_dofs, free_dofs] \ f[free_dofs]
    info("result vector")
    dump(u)
    =#

    u_disp = element1("displacement", [1.0, 1.0], 0.0)
    info("Displacement = $u_disp")

    @test isapprox(u_disp, expected)
end

