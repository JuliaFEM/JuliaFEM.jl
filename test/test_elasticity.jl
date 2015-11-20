# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

module ElasticityTests

using JuliaFEM.Test
using JuliaFEM: Seg2, Quad4, Field, FieldSet, CPS4,
                get_basis, solve!,
                PlaneStressElasticityProblem

function test_elasticity_volume_load()
    element = Quad4([1, 2, 3, 4])
    element["geometry"] = Vector[[0.0, 0.0], [10.0, 0.0], [10.0, 1.0], [0.0, 1.0]]
    element["youngs modulus"] = 500.0
    element["poissons ratio"] = 0.3
    element["displacement load"] = Vector[[0.0, -10.0], [0.0, -10.0], [0.0, -10.0], [0.0, -10.0]]
    free_dofs = [3, 4, 5, 6]
    problem = PlaneStressElasticityProblem()
    push!(problem, element)
    solve!(problem, free_dofs; max_iterations=10)
    disp = get_basis(element)("displacement", [1.0, 1.0])
    info("displacement at tip: $disp")
    # verified using Code Aster.
    @test isapprox(disp[2], -8.77303119819776)
end

function test_elasticity_surface_load()
    N = Vector[[0.0, 0.0], [10.0, 0.0], [10.0, 1.0], [0.0, 1.0]]

    element1 = Quad4([1, 2, 3, 4])
    element1["geometry"] = Vector[N[1], N[2], N[3], N[4]]
    element1["youngs modulus"] = 500.0
    element1["poissons ratio"] = 0.3
    element2 = Seg2([3, 4])
    element2["geometry"] = Vector[N[3], N[4]]
    element2["displacement traction force"] = Vector[[0.0, -10.0], [0.0, -10.0]]

    free_dofs = [3, 4, 5, 6]
    problem = PlaneStressElasticityProblem()
    push!(problem, element1)
    push!(problem, element2)
    solve!(problem, free_dofs; max_iterations=10)
    disp = get_basis(element1)("displacement", [1.0, 1.0])[2]
    info("displacement at tip: $disp")
    # verified using Code Aster.
    @test isapprox(disp, -9.33106637611714)
end

end
