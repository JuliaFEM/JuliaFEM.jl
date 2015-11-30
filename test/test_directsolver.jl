# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

module DirectSolverTests

using JuliaFEM.Test
using JuliaFEM

using JuliaFEM: Seg2, Quad4
using JuliaFEM: PlaneStressElasticityProblem, DirichletProblem
using JuliaFEM: DirectSolver

function test_solver_multiple_dirichlet_bc()

    N = Vector[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

    e1 = Quad4([1, 2, 4, 3])
    e1["geometry"] = Vector[N[1], N[2], N[4], N[3]]
    e1["youngs modulus"] = 900.0
    e1["poissons ratio"] = 0.25
    b1 = Seg2([3, 4])
    b1["geometry"] = Vector[N[3], N[4]]
    b1["displacement traction force"] = (
        0.0 => Vector[[0.0,    0.0], [0.0,    0.0]],
        1.0 => Vector[[0.0, -100.0], [0.0, -100.0]])

    problem = PlaneStressElasticityProblem()
    push!(problem, e1)
    push!(problem, b1)

    # manually solve problem 1
    # free_dofs = [3, 5, 6, 8]
    # free_dofs = [3, 6, 7, 8]
    #solve!(problem, free_dofs, 0.0; max_iterations=10)
    #disp = e1("displacement", [1.0, 1.0], 0.0)
    #info("displacement at tip: $disp")
    #@test isapprox(disp, [3.17431158889468E-02, -1.38591518927826E-01])

    # boundary elements for dirichlet dx=0
    dx = Seg2([1, 3])
    dx["geometry"] = Vector[N[1], N[3]]
    dx["displacement 1"] = 0.0

    # boundary elements for dirichlet dy=0
    dy = Seg2([1, 2])
    dy["geometry"] = Vector[N[1], N[2]]
    dy["displacement 2"] = 0.0
  
    problem2 = DirichletProblem("displacement", 2)
    push!(problem2, dx)

    problem3 = DirichletProblem("displacement", 2)
    push!(problem3, dy)

    solver = DirectSolver()
    push!(solver, problem)
    push!(solver, problem2)
    push!(solver, problem3)

    # launch solver
    norm = solver(0.0)
    norm = solver(1.0)
    disp = e1("displacement", [1.0, 1.0], 1.0)
    info("displacement at tip: $disp")
    @test isapprox(disp, [3.17431158889468E-02, -1.38591518927826E-01])

end
#test_solver_multiple_dirichlet_bc()


function test_solver_no_convergence()

    N = Vector[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

    e1 = Quad4([1, 2, 4, 3])
    e1["geometry"] = Vector[N[1], N[2], N[4], N[3]]
    e1["youngs modulus"] = 900.0
    e1["poissons ratio"] = 0.25
    b1 = Seg2([3, 4])
    b1["geometry"] = Vector[N[3], N[4]]
    b1["displacement traction force"] = Vector[[100.0, 100.0], [100.0, 100.0]]

    problem = PlaneStressElasticityProblem()
    push!(problem, e1)
    push!(problem, b1)

    # boundary elements for dirichlet dx=0
    dx = Seg2([1, 3])
    dx["geometry"] = Vector[N[1], N[3]]
    dx["displacement 1"] = 0.0

    # boundary elements for dirichlet dy=0
    dy = Seg2([1, 2])
    dy["geometry"] = Vector[N[1], N[2]]
    dy["displacement 2"] = 0.0
  
    problem2 = DirichletProblem("displacement", 2)
    push!(problem2, dx)

    problem3 = DirichletProblem("displacement", 2)
    push!(problem3, dy)

    solver = DirectSolver()
    solver.max_iterations = 1
    push!(solver, problem)
    push!(solver, problem2)
    push!(solver, problem3)

    # launch solver
    iterations, status = solver(0.0)

    @test status == false

end

function test_solver_multiple_bodies_multiple_dirichlet_bc()
    N = Vector[
        [0.0, 0.0], [1.0, 0.0],
        [0.0, 1.0], [1.0, 1.0],
        [0.0, 2.0], [1.0, 2.0]]

    e1 = Quad4([1, 2, 4, 3])
    e1["geometry"] = Vector[N[1], N[2], N[4], N[3]]
    e2 = Quad4([3, 4, 6, 5])
    e2["geometry"] = Vector[N[3], N[4], N[6], N[5]]
    for el in [e1, e2]
        el["youngs modulus"] = 900.0
        el["poissons ratio"] = 0.25
    end
    b1 = Seg2([5, 6])
    b1["geometry"] = Vector[N[5], N[6]]
    b1["displacement traction force"] = Vector[[0.0, -100.0], [0.0, -100.0]]

    body1 = PlaneStressElasticityProblem()
    push!(body1, e1)

    body2 = PlaneStressElasticityProblem()
    push!(body2, e2)
    push!(body2, b1)

    # boundary elements for dirichlet dx=0
    dx1 = Seg2([1, 3])
    dx1["geometry"] = Vector[N[1], N[3]]
    dx2 = Seg2([3, 5])
    dx2["geometry"] = Vector[N[3], N[5]]
    for dx in [dx1, dx2]
        dx["displacement 1"] = 0.0
    end

    boundary1 = DirichletProblem("displacement", 2)
    push!(boundary1, dx1)
    push!(boundary1, dx2)

    # boundary elements for dirichlet dy=0
    dy1 = Seg2([1, 2])
    dy1["geometry"] = Vector[N[1], N[2]]
    dy1["displacement 2"] = 0.0
  
    boundary2 = DirichletProblem("displacement", 2)
    push!(boundary2, dy1)


    solver = DirectSolver()
    push!(solver, body1)
    push!(solver, body2)
    push!(solver, boundary1)
    push!(solver, boundary2)

    # launch solver
    norm = solver(0.0)

    disp = e2("displacement", [1.0, 1.0], 0.0)
    info("displacement at tip: $disp")
    # code aster verification, two_elements.comm
    @test isapprox(disp, [3.17431158889468E-02, -2.77183037855653E-01])

end

# test_solver_multiple_bodies_multiple_dirichlet_bc()

end
