# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md


using Base.Test

using JuliaFEM: Quad4, Field, FieldSet, CPS4, get_basis, solve!, PlaneStressElasticityProblem


function run()
    element = Quad4([1, 2, 3, 4])
    element["geometry"] = Vector[[0.0, 0.0], [10.0, 0.0], [10.0, 1.0], [0.0, 1.0]]
    element["youngs modulus"] = 500.0
    element["poissons ratio"] = 0.3
    element["displacement load"] = Vector[[0.0, -10.0], [0.0, -10.0], [0.0, -10.0], [0.0, -10.0]]
    equation = CPS4(element)
    free_dofs = [3, 4, 5, 6]
    problem = PlaneStressElasticityProblem([equation])
    solve!(problem, free_dofs; max_iterations=10)
    #solve!(equation, "displacement", free_dofs; max_iterations=10)
    disp = get_basis(element)("displacement", [1.0, 1.0])[2]
    info("displacement at tip: $disp")
    # verified using Code Aster.
    @test disp ≈ -8.77303119819776
end

run()
