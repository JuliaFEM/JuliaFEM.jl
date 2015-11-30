# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

module ElasticityTests

using JuliaFEM.Test
using JuliaFEM.Core: Seg2, Quad4, PlaneStressElasticityProblem, solve!

function test_elasticity_volume_load()
    element = Quad4([1, 2, 3, 4])
    element["geometry"] = Vector[[0.0, 0.0], [10.0, 0.0], [10.0, 1.0], [0.0, 1.0]]
    element["youngs modulus"] = 500.0
    element["poissons ratio"] = 0.3
    element["displacement load"] = Vector[[0.0, -10.0], [0.0, -10.0], [0.0, -10.0], [0.0, -10.0]]
    element["displacement"] = (0.0 => Vector{Float64}[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    problem = PlaneStressElasticityProblem()
    push!(problem, element)

    free_dofs = [3, 4, 5, 6]
    solve!(problem, free_dofs, 0.0; max_iterations=10)
    disp = element("displacement", [1.0, 1.0], 0.0)
    ip1 = last(element["integration points"])[1]
    ip2 = last(element["integration points"])[2]
    strain = ip1["gl strain"]
    info("displacement at tip: $disp")
    info("strain in first ip: $strain. ip coord = $(ip1.xi) and weight = $(ip1.weight)")
    # verified using Code Aster, verification/2015-10-22-plane-stress/cplan_grot_gdep_volume_force.resu
    @test isapprox(disp[2], -8.77303119819776)
end
#test_elasticity_volume_load()

function test_elasticity_surface_load()
    N = Vector[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

    element1 = Quad4([1, 2, 4, 3])
    element1["geometry"] = Vector[N[1], N[2], N[4], N[3]]
    element1["youngs modulus"] = 900.0
    element1["poissons ratio"] = 0.25
    element1["displacement"] = (0.0 => Vector{Float64}[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

    element2 = Seg2([3, 4])
    element2["geometry"] = Vector[N[3], N[4]]
    element2["displacement traction force"] = Vector[[0.0, -100.0], [0.0, -100.0]]
    element2["displacement"] = (0.0 => Vector{Float64}[[0.0, 0.0], [0.0, 0.0]])

    #free_dofs = [3, 5, 6, 8]
    free_dofs = [3, 6, 7, 8]
    problem = PlaneStressElasticityProblem()
    push!(problem, element1)
    push!(problem, element2)
    solve!(problem, free_dofs, 0.0; max_iterations=10)
    disp = element1("displacement", [1.0, 1.0], 0.0)
    info("displacement at tip: $disp")
    # verified using Code Aster.
    @test isapprox(disp, [3.17431158889468E-02, -1.38591518927826E-01])
end

end
