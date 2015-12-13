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
#    element1["youngs modulus"] = 900.0
#    element1["poissons ratio"] = 0.25
    element1["youngs modulus"] = 900.0
    element1["poissons ratio"] = 0.25
    element1["displacement"] = (0.0 => Vector{Float64}[[0.0, 0.0, 0.0] for i=1:8])

    element2 = Quad4([5, 6, 7, 8])
    set_geometry!(element2, nodes)
    element2["displacement traction force"] = Vector{Float64}[[0.0, 0.0, -100.0] for i=1:4]
    element2["displacement"] = (0.0 => Vector{Float64}[[0.0, 0.0, 0.0] for i=1:4])

    problem = ElasticityProblem()
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
    ass = JuliaFEM.Core.assemble(problem, 0.0)
    info(reshape(full(ass.force_vector), 3, 8))
    info("initial stiffness matrix")
    dump(round(Int, full(ass.stiffness_matrix))[free_dofs, free_dofs])
    solve!(problem, free_dofs, 0.0; max_iterations=10)

#=
    dx = Quad4([1, 4, 8, 5])
    dx["displacement 1"] = 0.0
    dy = Quad4([1, 5, 6, 2])
    dy["displacement 2"] = 0.0
    dz = Quad4([1, 2, 3, 4])
    dz["displacement 3"] = 0.0
    bc = DirichletProblem("displacement", 3)
    for el in [dx, dy, dz]
        set_geometry!(el, nodes)
        push!(bc, el)
    end
    solver = JuliaFEM.Core.DirectSolver()
    push!(solver, problem)
    push!(solver, bc)
    solver.dump_matrices = true
    solver.name = "3d_hex8"
    solver(0.0)
=#

    disp = element1("displacement", [1.0, 1.0, 1.0], 0.0)
    info("displacement at tip: $disp")
    info("displacement on element: ")
    for (i, d) in enumerate(element1("displacement", 0.0))
        @printf "%d % f % f % f\n" [i;d]...
    end
    # verified using Code Aster.
    # 2015-12-12-continuum-elasticity/vim c3d_grot_gdep_traction_force.comm
    @test isapprox(disp, [3.17431158889468E-02, 3.17431158889468E-02, -1.38591518927826E-01])
    #@test isapprox(disp, [2.80559539222183E-03, 2.80559539222183E-03, -1.13019918093242E-02])
end
#test_continuum_elasticity_with_surface_load()

end
