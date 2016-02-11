# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM.Test
using JuliaFEM.Core: Node, update!, Quad4, Seg2, Problem, Elasticity, Solver, Dirichlet
using JuliaFEM.Preprocess: aster_parse_nodes

@testset "test 2d linear elasticity with surface load" begin

    nodes = Dict{Int64, Node}(
        1 => [0.0, 0.0],
        2 => [1.0, 0.0],
        3 => [1.0, 1.0],
        4 => [0.0, 1.0])

    E = 288.0
    nu = 1.0/3.0
    f = -E/10.0
    expected = f/E*[-nu, 1]

    # field problem
    element1 = Quad4([1, 2, 3, 4])
    element2 = Seg2([3, 4])
    update!([element1, element2], "geometry", nodes)
    update!(element1, "youngs modulus", E)
    update!(element1, "poissons ratio", nu)
#   update!(element2, "displacement traction force", [0.0, f])
    update!(element2, "displacement traction force", Vector{Float64}[[0.0, f], [0.0, f]])
    # type, name, dimension
    elasticity_problem = Problem(Elasticity, "block", 2)
    elasticity_problem.properties.formulation = :plane_stress
    push!(elasticity_problem, element1, element2)

    # boundary condition, displacement symmetry
    sym13 = Seg2([1, 2])
    sym23 = Seg2([4, 1])
    update!([sym13, sym23], "geometry", nodes)
    update!(sym13, "displacement 2", 0.0)
    update!(sym23, "displacement 1", 0.0)
    # type, name, dimension, unknown_field_name
    boundary_problem = Problem(Dirichlet, "symmetry boundaries", 2, "displacement")
    push!(boundary_problem, sym13, sym23)
    
    solver = Solver("solve block problem")
    solver.is_linear_system = true # to get linear solution
    push!(solver, elasticity_problem)
    push!(solver, boundary_problem)
    call(solver)
    element1 = elasticity_problem.elements[1]
    u_disp = element1("displacement", [1.0, 1.0], 0.0)
    info("Displacement = $u_disp")
    @test isapprox(u_disp, expected)

end

@testset "test 2d nonlinear elasticity with surface load" begin
    nodes = Dict{Int64, Node}(
        1 => [0.0, 0.0],
        2 => [1.0, 0.0],
        3 => [0.0, 1.0],
        4 => [1.0, 1.0])
    element1 = Quad4([1, 2, 4, 3])
    update!(element1, "geometry", nodes)
    element1["youngs modulus"] = 900.0
    element1["poissons ratio"] = 0.25
    element2 = Seg2([3, 4])
    update!(element2, "geometry", nodes)
    element2["displacement traction force"] = Vector[[0.0, -100.0], [0.0, -100.0]]
    elasticity_problem = Problem(Elasticity, "bock", 2)
    elasticity_problem.properties.formulation = :plane_stress
    push!(elasticity_problem, element1, element2)

    # boundary condition, displacement symmetry
    sym13 = Seg2([1, 2])
    sym23 = Seg2([3, 1])
    update!([sym13, sym23], "geometry", nodes)
    update!(sym13, "displacement 2", 0.0)
    update!(sym23, "displacement 1", 0.0)
    # type, name, dimension, unknown_field_name
    boundary_problem = Problem(Dirichlet, "symmetry boundaries", 2, "displacement")
    push!(boundary_problem, sym13, sym23)

    solver = Solver("solve block problem")
    push!(solver, elasticity_problem)
    push!(solver, boundary_problem)
    call(solver)
    element1 = elasticity_problem.elements[1]

    u_disp = element1("displacement", [1.0, 1.0], 0.0)
    # verified using Code Aster.
    u_expected = [3.17431158889468E-02, -1.38591518927826E-01]
    info("Displacement = $u_disp")
    @test isapprox(u_disp, u_expected)
end

@testset "test continuum linear elasticity with surface load" begin

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

