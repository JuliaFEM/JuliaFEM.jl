# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

module APITests

using JuliaFEM.Test

using JuliaFEM.Preprocess: parse_abaqus
using JuliaFEM.API: Model, Element, ElementSet, Material, LoadCase,
DirichletBC, NeumannBC, add_boundary_condition!, add_solver!, add_material!
using JuliaFEM.Interfaces: solve!


function test_basic()
    # basic workflow, copied from test_solver.jl
    model = Model("Piston Calculation")

    model.nodes[1] = [0.0, 0.0]
    model.nodes[2] = [1.0, 0.0]
    model.nodes[3] = [1.0, 1.0]
    model.nodes[4] = [0.0, 1.0]
    
    # create elements
    e1 = Element(1, [1, 2, 3, 4], :Quad4)
    e2 = Element(2, [1, 2], :Seg2)
    e3 = Element(3, [3, 4], :Seg2)
    model.elements[1] = e1  
    model.elements[2] = e2  
    model.elements[3] = e3  


    # element set
    elset = ElementSet("body", [e1, e2])
    elset2 = ElementSet("heat_flux", [e2]) 
    elset3 = ElementSet("constant_temp", [e3])
    elset4 = ElementSet("set_material", [e1]) 

    model.elsets["body"] = elset
    model.elsets["heat_flux"] = elset2
    model.elsets["constant_temp"] = elset3
    model.elsets["set_material"] = elset4

    # material properties
    material = Material("MatMat")
    material["temperature thermal conductivity"] = 6.0
    material["density"] = 36.0
    add_material!(model, "set_material", material)

    # Create problem
    field_problem = LoadCase(:HeatProblem)
    field_problem.sets = "body"

    # boundary conditions
    bc = DirichletBC("constant_temp", "temperature" => 0.0)
    ne = NeumannBC("heat_flux", "temperature flux" => ((0.0 =>   0.0),
                                                  (1.0 => 600.0)))
    
    # LoadCase
    add_boundary_condition!(field_problem, bc)
    add_boundary_condition!(field_problem, ne)

    # Get solver
    add_solver!(field_problem, :LinearSolver)

    # add case
    model.load_cases["Heat problem"] = field_problem

    # Solve problem
    solve!(model, "Heat problem", 1.0)
    xi = [0.0, -1.0]
    T = model.elements[1].results("temperature", xi, 1.0)
    X = model.elements[2].results("geometry", xi, 1.0)
    info("Temperature at point X = $X is T = $T")
    @test isapprox(T, 100.0)
end

function test_piston_8789()
    abaqus_input = open(parse_abaqus, "../geometry/piston/piston_8789_P1.inp")

    model = Model("Piston Calculation", abaqus_input)
end

end
