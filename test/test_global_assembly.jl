# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

module AssemblyTests

using JuliaFEM.Test
using JuliaFEM: Quad4, Seg2, FieldSet, Field, HeatProblem
using JuliaFEM: Assembly, assemble!

"""assemble a simple two element problem and solve"""
function test_assembly()
    info("create elements")
    el1 = Quad4([1, 2, 3, 4])
    el1["geometry"] = Vector[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    el1["temperature thermal conductivity"] = 6.0
    el1["temperature load"] = 12.0
    el1["density"] = 36.0
    el2 = Seg2([1, 2])
    el2["geometry"] = Vector[[0.0, 0.0], [1.0, 0.0]]
    # Boundary load, linear ramp 0 -> 600 at time 0 -> 1
    el2["temperature flux"] = ((0.0 => 0.0), (1.0 => 600.0))
    info("element created")

    problem = HeatProblem()
    info("problem created. pushing elements")
    push!(problem, el1)
    push!(problem, el2)

    info("creating assembly from equations")
    assembly = Assembly()
    assemble!(assembly, problem, 1.0)
    info("solving")

    free_dofs = [1, 2]
    A = full(assembly.stiffness_matrix)[free_dofs, free_dofs]
    b = full(assembly.force_vector)[free_dofs]
    u = A \ b
    info("solution u=$u")
    @test isapprox(u, [101.0, 101.0])
end

end
