# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM: Quad4, Seg2, FieldSet, Field, PlaneHeatProblem
using JuliaFEM: initialize_global_assembly, calculate_global_assembly!
using FactCheck

facts("assemble a simple two element problem and solve") do 
    el1 = Quad4([1, 2, 3, 4])
    el1["geometry"] = Vector[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    el1["temperature thermal conductivity"] = 6.0
    el1["temperature load"] = [12.0, 12.0, 12.0, 12.0]
    el1["density"] = 10

    el2 = Seg2([1, 2])
    el2["geometry"] = Vector[[0.0, 0.0], [1.0, 0.0]]

    # Boundary load, linear ramp 0 -> 600 at time 0 -> 1
    el2["temperature flux"] = FieldSet(Field[Field(0.0, 0.0), Field(1.0, 600.0)])

    problem = PlaneHeatProblem()
    push!(problem, el1)
    push!(problem, el2)

    global_assembly = initialize_global_assembly(problem)
    calculate_global_assembly!(global_assembly, problem)
    free_dofs = [1, 2]
    A = lufact(global_assembly.stiffness_matrix[free_dofs, free_dofs])
    b = full(global_assembly.force_vector)[free_dofs]
    u = A \ b
    @fact u --> roughly([101.0, 101.0])
end
