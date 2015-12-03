# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

module APITests

using JuliaFEM.Test

using JuliaFEM.Preprocess: parse_abaqus
using JuliaFEM.API: Model, Element, ElementSet, Material, LoadCase,
DirichletBC, NeumannBC, add_boundary_condition!, add_solver!, add_material!,
add_node!, add_element!, add_element_set!, add_load_case!
using JuliaFEM.Interfaces: solve!


function test_basic()
    # basic workflow, copied from test_solver.jl
    model = Model("Piston Calculation")

    add_node!(model, 1, [0.0, 0.0]) 
    model.nodes[2] = [1.0, 0.0]
    add_node!(model, 3, [1.0, 1.0])
    model.nodes[4] = [0.0, 1.0]
    
    # create elements
    e1 = Element(1, [1, 2, 3, 4], :Quad4)
    e2 = Element(2, [1, 2], :Seg2)
    e3 = Element(3, [3, 4], :Seg2)
    model.elements[1] = e1  
    model.elements[2] = e2  
    add_element!(model, 3, :Seg2, [3, 4])

    # element set
    elset = ElementSet("body", [e1, e2])
    elset4 = ElementSet("set_material", [e1]) 

    model.elsets["body"] = elset
    add_element_set!(model, "heat_flux", [2])
    add_element_set!(model, "constant_temp", [e3])
    add_element_set!(model, elset4)

    # material properties
    material = Material("myMaterial")
    material["temperature thermal conductivity"] = 6.0
    material["density"] = 36.0
    add_material!(model, "set_material", material)

    # Create problem
    field_problem = LoadCase(:HeatProblem)
    add_element_set!(field_problem, "body") # is this necessary?

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
    add_load_case!(model, "Heat problem", field_problem)
    #model.load_cases["Heat problem"] = field_problem

    # Solve problem
    solve!(model, "Heat problem", 1.0)
    xi = [0.0, -1.0]
    T = model.elements[1].results("temperature", xi, 1.0)
    X = model.elements[2].results("geometry", xi, 1.0)
    info("Temperature at point X = $X is T = $T")
    @test isapprox(T, 100.0)
end

function test_piston_8789()
    abaqus_input = open(parse_abaqus, "./geometry/piston/piston_8789_P1.inp")

    model = Model("Piston Calculation", abaqus_input)
    @test length(keys(model.elsets)) == 4
    @test length(keys(model.nsets)) == 1
    @test length(keys(model.elements)) == 37331
    @test length(keys(model.nodes)) == 8789
end

#function test_piston_107168()
#    abaqus_input = open(parse_abaqus, "./geometry/piston/piston_107168_P2.inp")
#
#    model = Model("Piston Calculation", abaqus_input)
#    @test length(keys(model.elsets)) == 3
#    @test length(keys(model.nsets)) == 1
#    @test length(keys(model.elements)) == 65948
#    @test length(keys(model.nodes)) == 107168
#end 

#test_piston_107168()

end
