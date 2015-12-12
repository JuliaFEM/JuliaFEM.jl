# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

module LinearElasticityTests

using JuliaFEM
using JuliaFEM.Test

using JuliaFEM.Core: Seg2, Quad4, Hex8, LinearElasticityProblem, get_connectivity,
                     assemble, PlaneStressLinearElasticityProblem
using JuliaFEM.Preprocess: aster_parse_nodes


function test_plane_stress_linear_elasticity_with_surface_load()
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

    problem = PlaneStressLinearElasticityProblem()
    push!(problem, element1)
    push!(problem, element2)

    free_dofs = Int64[3, 5, 6, 8]

    info("initial force vector")
    ass = assemble(problem, 0.0)
    f = full(ass.force_vector)
    K = full(ass.stiffness_matrix)
    dump(reshape(f, 2, 4))
    info("initial stiffness matrix")
    dump(round(Int, K)[free_dofs, free_dofs])

    u = zeros(2, 4)
    u[free_dofs] = K[free_dofs, free_dofs] \ f[free_dofs]

    info("result vector")
    dump(u)
    # verified using Code Aster.
    # 2015-10-22-plane-stress/cplan_linear_traction_force.*
    @test isapprox(u[:,3], [2.77777777777778E-03, -1.11111111111111E-02])
end
#test_plane_stress_linear_elasticity_with_surface_load()


function test_continuum_elasticity_with_surface_load()
    nodes = JuliaFEM.Preprocess.aster_parse_nodes("""
    COOR_3D
    N1          0.0 0.0 0.0
    N2          1.0 0.0 0.0
    N3          1.0 1.0 0.0
    N4          0.0 1.0 0.0
    N5          0.0 0.0 1.0
    N6          1.0 0.0 1.0
    N7          1.0 1.0 1.0
    N8          0.0 1.0 1.0
    FINSF
    """)

    function set_geometry!(element, nodes)
        element["geometry"] = Vector{Float64}[nodes[i] for i in get_connectivity(element)]
    end
    element1 = Hex8([1, 2, 3, 4, 5, 6, 7, 8])
    set_geometry!(element1, nodes)
    element1["youngs modulus"] = 9000.0
    element1["poissons ratio"] = 0.25

    element2 = Quad4([5, 6, 7, 8])
    set_geometry!(element2, nodes)
    element2["displacement traction force"] = Vector{Float64}[[0.0, 0.0, -100.0] for i=1:4]

    problem = LinearElasticityProblem()
    push!(problem, element1)
    push!(problem, element2)

    free_dofs = zeros(Bool, 8, 3)
    x = 1
    y = 2
    z = 3
    free_dofs[2, x] = true
    free_dofs[3, [x, y]] = true
    free_dofs[4, y] = true
    free_dofs[5, z] = true
    free_dofs[6, [x, z]] = true
    free_dofs[7, [x, y, z]] = true
    free_dofs[8, [y, z]] = true
    free_dofs = find(vec(free_dofs'))
    info("free dofs: $free_dofs")

    info("initial force vector")
    ass = assemble(problem, 0.0)
    f = full(ass.force_vector)
    K = full(ass.stiffness_matrix)
    dump(reshape(f, 3, 8))
    info("initial stiffness matrix")
    dump(round(Int, K)[free_dofs, free_dofs])

    u = zeros(3, 8)
    u[free_dofs] = K[free_dofs, free_dofs] \ f[free_dofs]

    info("result vector")
    dump(u)
    # verified using Code Aster.
    # 2015-12-12-continuum-elasticity/c3d_linear.*
    @test isapprox(u[:,7], [2.77777777777778E-02, 2.77777777777778E-02, -1.11111111111111E-01])
end
#test_continuum_elasticity_with_surface_load()


end
