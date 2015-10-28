# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# unit tests for heat equations

using FactCheck
using JuliaFEM: Quad4, Field, FieldSet, CPS4, get_basis, solve!, PlaneStressElasticityProblem


facts("test plane elasticity on single element, volume load") do
    element = Quad4([1, 2, 3, 4])
    element["geometry"] = FieldSet(Field(Vector[[0.0, 0.0], [10.0, 0.0], [10.0, 1.0], [0.0, 1.0]]))
    element["youngs modulus"] = FieldSet(Field(500.0))
    element["poissons ratio"] = FieldSet(Field(0.3))
    element["displacement load"] = FieldSet(Field(0.0, Vector[[0.0, -10.0], [0.0, -10.0], [0.0, -10.0], [0.0, -10.0]]))
    equation = CPS4(element)
    free_dofs = [3, 4, 5, 6]
    problem = PlaneStressElasticityProblem([equation])
    solve!(problem, free_dofs; max_iterations=10)
    #solve!(equation, "displacement", free_dofs; max_iterations=10)
    disp = get_basis(element)("displacement", [1.0, 1.0])[2]
    Logging.info("displacement at tip: $disp")
    # verified using Code Aster.
    @fact disp --> roughly(-8.77303119819776E+00)
end

