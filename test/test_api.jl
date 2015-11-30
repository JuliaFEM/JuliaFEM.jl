# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

module APITests

using JuliaFEM
using JuliaFEM.Test
using JuliaFEM: Model, Node, Element, ElementSet,
Material, Quad4, Seg2 

function test_basic()
    # basic workflow, copied from test_solver.jl
    model = Model("my calculation model")

    # create nodes
    model.nodes[1] = [0.0, 0.0]
    model.nodes[2] = [1.0, 0.0]
    model.nodes[3] = [1.0, 1.0]
    model.nodes[4] = [0.0, 1.0]
    
    # create elements
    e1 = Element{Quad4}([1, 2, 3, 4])
    e2 = Element{Seg2}([1, 2])

    # element set
    elset = ElementSet("body", e1, e2)
    elset2 = ElementSet("boundary", e2) 
    # push!(model, elset, elset2)

    # material properties
   # material = Material("my material")
   # material["temperature thermal conductivity"] = 6.0
   # material["density"] = 36.0
   # set_material!(model, material, "Es")

    # Create problem
   # field_problem = HeatProblem()

    # boundary conditions
  #  bc = DirichletBC("body", "temperature" => 0.0)
    # bc = DirichletBC("Es", 0.0)
  #  ne = NeumannBC("boundary", "temperature flux" => ((0.0 => 0.0),
  #                                                    (1.0=>600.0)))
    
    # LoadCase
  #  case = LoadCase("Heat problem")
  #  push!(case, field_problem)
  #  push!(case, bc, ne)

    # Get solver
  #  solver = LinearSolver
  #  push!(case, solver)

    # add case
  #  push!(model, case)

    # Solve problem
  #  solve!(model, 1.0, "Heat problem")
  #  xi = [0.0, -1.0]
  #  T = el1("temperature", xi, 1.0)
    # T = model("temperature", [0.5, 0.0], 1)
end

test_basic()
end
