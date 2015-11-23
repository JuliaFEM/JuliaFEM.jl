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
    solve!(problem, free_dofs, 0.0; max_iterations=10)
    disp = get_basis(element)("displacement", [1.0, 1.0], 0.0)
    info("displacement at tip: $disp")
    # verified using Code Aster.
    @test isapprox(disp[2], -8.77303119819776)
end

function test_elasticity_surface_load()
    N = Vector[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

    element1 = Quad4([1, 2, 4, 3])
    element1["geometry"] = Vector[N[1], N[2], N[4], N[3]]
    element1["youngs modulus"] = 900.0
    element1["poissons ratio"] = 0.25
    element2 = Seg2([3, 4])
    element2["geometry"] = Vector[N[3], N[4]]
    element2["displacement traction force"] = Vector[[0.0, -100.0], [0.0, -100.0]]

    #free_dofs = [3, 5, 6, 8]
    free_dofs = [3, 6, 7, 8]
    problem = PlaneStressElasticityProblem()
    push!(problem, element1)
    push!(problem, element2)
    solve!(problem, free_dofs, 0.0; max_iterations=10)
    #disp = get_basis(element1)("displacement", [1.0, 1.0], 1.0)[2]
    info(last(element1["displacement"]))
    disp = element1("displacement", [1.0, 1.0], 0.0)
    info("displacement at tip: $disp")
    # verified using Code Aster.
    @test isapprox(disp, [3.17431158889468E-02, -1.38591518927826E-01])
end

end
