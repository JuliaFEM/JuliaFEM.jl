# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

module SolverTests

using JuliaFEM.Test
using JuliaFEM

using JuliaFEM: DirichletProblem, Seg2, PlaneHeatProblem, Quad4, SimpleSolver, get_element, get_basis, MortarElement, MortarProblem, PlaneStressElasticityProblem, solve!, DirectSolver

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
    el3["temperature"] = 0.0
    problem2 = DirichletProblem("temperature", 1)
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
#test_simplesolver()

function atest_direct_solver()

    N = Dict{Int, Vector}(
        1 => [0.0, 0.0],
        2 => [2.0, 0.0],
        3 => [4.0, 0.0],
        4 => [0.0, 1.0],
        5 => [2.0, 1.0],
        6 => [4.0, 1.0],
        7 => [0.0, 1.0],
        8 => [1.0, 1.0],
        9 => [3.0, 1.0],
        10 => [4.0, 1.0],
        11 => [0.0, 2.0],
        12 => [1.0, 2.0],
        13 => [3.0, 2.0],
        13 => [4.0, 1.0])

   # volume elements
   e1 = Quad4([1, 2, 5, 4])
   e1["geometry"] = Vector[N[1], N[2], N[5], N[4]]
   e2 = Quad4([2, 3, 6, 5])
   e2["geometry"] = Vector[N[2], N[3], N[6], N[5]]
   e3 = Quad4([7, 8, 12, 11])
   e3["geometry"] = Vector[N[7], N[8], N[12], N[11]]
   e4 = Quad4([8, 9, 13, 12])
   e4["geometry"] = Vector[N[8], N[9], N[13], N[12]]
   e5 = Quad4([9, 10, 14, 13])
   e5["geometry"] = Vector[N[9], N[10], N[14], N[13]]

   # boundary elements for boundary load
   b1 = Seg2([11, 12])
   b1["geometry"] = Vector[N[11], N[12]]
   b1["displacement traction force"] = Vector[[0.0, -10.0], [0.0, -10.0]]
   b2 = Seg2([12, 13])
   b2["geometry"] = Vector[N[12], N[13]]
   b2["displacement traction force"] = Vector[[0.0, -10.0], [0.0, -10.0]]
   b3 = Seg3([13, 14])
   b3["geometry"] = Vector[N[13], N[14]]
   b3["displacement traction force"] = Vector[[0.0, -10.0], [0.0, -10.0]]

   # boundary elements for dirichlet dy=0
   d1 = Seg2([1, 2])
   d1["geometry"] = Vector[N[1], N[2]]
   d1["displacement 2"] = 0.0
   d2 = Seg2([2, 3])
   d2["geometry"] = Vector[N[2], N[3]]
   d2["displacement 2"] = 0.0
  
   # boundary elements for dirichlet dx=0
   d3 = Seg2([1, 4])
   d3["geometry"] = Vector[N[1], N[4]]
   d3["displacement 1"] = 0.0
   d4 = Seg2([4, 11])
   d4["geometry"] = Vector[N[4], N[11]]
   d4["displacmeent 1"] = 0.0

   # mortar elements to tie meshes -- masters
   m1 = MSeg2([4, 5])
   m1["geometry"] = Vector[N[4], N[5]]
   m2 = MSeg2([5, 6])
   m2["geometry"] = Vector[N[5], N[6]]

   # mortar elements to tie meshes -- slaves
   rotation_matrix(phi) = [cos(phi) -sin(phi); sin(phi) cos(phi)]
   phi = rotation_matrix(-pi/2)
   m3 = MSeg2([7, 8])
   m3["geometry"] = Vector[N[7], N[8]]
   m3["nodal ntsys"] = Matrix[phi, phi]
   m3["master elements"] = MortarElement[m1, m2]
   m4 = MSeg2([8, 9])
   m4["geometry"] = Vector[N[8], N[9]]
   m4["nodal ntsys"] = Matrix[phi, phi]
   m4["master elements"] = MortarElement[m1, m2]
   m5 = MSeg2([9, 10])
   m5["geometry"] = Vector[N[9], N[10]]
   m5["nodal ntsys"] = Matrix[phi, phi]
   m5["master elements"] = MortarElement[m1, m2]

   problem1 = PlaneStressElasticityProblem()
   push!(problem1, e1)
   push!(problem1, e2)
   push!(problem1, e3)
   push!(problem1, e4)
   push!(problem1, e5)
   push!(problem1, b1)
   push!(problem1, b2)
   push!(problem1, b3)

   problem2 = DirichletProblem()
   push!(problem2, d1)
   push!(problem2, d2)
   push!(problem2, d3)
   push!(problem2, d4)

   problem3 = MortarProblem()
   push!(problem3, m1)
   push!(problem3, m2)
   push!(problem3, m3)
   push!(problem3, m4)
   push!(problem3, m5)

   solver = DirectSolver()
   push!(solver, problem1)
   push!(solver, problem2)
   push!(solver, problem3)

   call(solver)

end

function test_solver_multiple_dirichlet_bc()
    N = Vector[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

    e1 = Quad4([1, 2, 4, 3])
    e1["geometry"] = Vector[N[1], N[2], N[4], N[3]]
    e1["youngs modulus"] = 900.0
    e1["poissons ratio"] = 0.25
    b1 = Seg2([3, 4])
    b1["geometry"] = Vector[N[3], N[4]]
    b1["displacement traction force"] = Vector[[0.0, -100.0], [0.0, -100.0]]

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

#   info(e1("displacement"))
#   info(last(e1["displacement"]))
    disp = e1("displacement", [1.0, 1.0], 0.0)
    info("displacement at tip: $disp")
    @test isapprox(disp, [3.17431158889468E-02, -1.38591518927826E-01])

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
