# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

module LinearElasticityTests

using JuliaFEM
using JuliaFEM.Test

using JuliaFEM.Core: Seg2, Quad4, Hex8, LinearElasticityProblem, get_connectivity,
                     assemble, PlaneStressLinearElasticityProblem, DirichletProblem,
                     LinearSolver
using JuliaFEM.Preprocess: aster_parse_nodes
using JuliaFEM.Core: PlaneStressLinearElasticPlasticProblem

function test_plane_stress_linear_elasticplastic_with_surface_load()
    nodes = Dict{Int64, Vector{Float64}}(
        1 => [0.0, 0.0],
        2 => [1.0, 0.0],
        3 => [1.0, 1.0],
        4 => [0.0, 1.0])

    function set_geometry!(element, nodes)
        element["geometry"] = Vector{Float64}[nodes[i] for i in get_connectivity(element)]
    end
    element1 = Quad4([1, 2, 3, 4])
    set_geometry!(element1, nodes)
    element1["youngs modulus"] = 9000.0
    element1["poissons ratio"] = 0.25

    element2 = Seg2([3, 4])
    set_geometry!(element2, nodes)
    element2["displacement traction force"] = Vector{Float64}[[0.0, -100.0] for i=1:2]

   # problem = PlaneStressLinearElasticityProblem()
    problem = PlaneStressLinearElasticPlasticProblem()
    push!(problem, element1)
    push!(problem, element2)

    free_dofs = Int64[3, 5, 6, 8]

    ass = assemble(problem, 0.0)
    f = full(ass.force_vector)
    K = full(ass.stiffness_matrix)
#   info("initial force vector")
#   dump(reshape(f, 2, 4))
#   info("initial stiffness matrix")
#   dump(round(Int, K)[free_dofs, free_dofs])

    u = zeros(2, 4)
    u[free_dofs] = K[free_dofs, free_dofs] \ f[free_dofs]

    info("result vector")
    dump(u)
    # verified using Code Aster.
    # 2015-10-22-plane-stress/cplan_linear_traction_force.*
    @test isapprox(u[:,3], [2.77777777777778E-03, -1.11111111111111E-02])
end
# test_plane_stress_linear_elasticplastic_with_surface_load()




end
