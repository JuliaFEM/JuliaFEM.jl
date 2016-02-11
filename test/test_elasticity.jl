# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

module ElasticityTests

using JuliaFEM.Test
using JuliaFEM
using JuliaFEM.Core: Seg2, Quad4, Hex8,
                     ElasticityProblem, PlaneStressElasticityProblem,
                     solve!, get_connectivity, DirichletProblem


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
    # function get_previous_ip(element::Element, current_ip::IntegrationPoint)
    # end
    # ipdata = element("integration points", time) => IntegrationPoint[ip1, ip2, ..., ipN]
    # for some_ip in ipdata
    #   if isapprox(some_ip.xi, ip.xi)
    #       info("found")
    #       last_value = some_ip("material parameter", time)
    #       break
    #   end
    # end
    #ip1 = last(element["integration points"])[1]
    #ip2 = last(element["integration points"])[2]
    # strain = ip1("gl strain")
    info("displacement at tip: $disp")
    #info("strain in first ip: $strain. ip coord = $(ip1.xi) and weight = $(ip1.weight)")
    # verified using Code Aster, verification/2015-10-22-plane-stress/cplan_grot_gdep_volume_force.resu
    @test isapprox(disp[2], -8.77303119819776)
end
#test_elasticity_volume_load()

end
