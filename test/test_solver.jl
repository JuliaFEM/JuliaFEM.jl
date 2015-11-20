# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

module SolverTests

using JuliaFEM.Test
using JuliaFEM

using JuliaFEM: DirichletProblem, Seg2, PlaneHeatProblem, Quad4, SimpleSolver, get_element, get_basis, MortarElement, MortarProblem, DirectSolver, PlaneStressElasticityProblem

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

function test_direct_solver()

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
   e1["geometry"] = Vector[N[1], N[2], N[3], N[4]]
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

end
