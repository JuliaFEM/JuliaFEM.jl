# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM.Test
using JuliaFEM.Core: Node, update!, Quad4, Seg2, Hex8, Problem, Elasticity, Solver, Dirichlet
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

    nodes = Dict{Int64, Node}(
    1 => [0.0, 0.0, 0.0],
    2 => [1.0, 0.0, 0.0],
    3 => [1.0, 1.0, 0.0],
    4 => [0.0, 1.0, 0.0],
    5 => [0.0, 0.0, 1.0],
    6 => [1.0, 0.0, 1.0],
    7 => [1.0, 1.0, 1.0],
    8 => [0.0, 1.0, 1.0])

    element1 = Hex8([1, 2, 3, 4, 5, 6, 7, 8])
    element2 = Quad4([5, 6, 7, 8])
    update!([element1, element2], "geometry", nodes)
    update!([element1], "youngs modulus", 900.0)
    update!([element1], "poissons ratio", 0.25)
    update!([element2], "displacement traction force", Vector{Float64}[[0.0, 0.0, -100.0] for i=1:4])

    elasticity_problem = Problem(Elasticity, "solve continuum block", 3)
    push!(elasticity_problem, element1)
    push!(elasticity_problem, element2)

    symxy = Quad4([1, 2, 3, 4])
    symxz = Quad4([1, 2, 6, 5])
    symyz = Quad4([1, 4, 8, 5])
    update!([symxy, symxz, symyz], "geometry", nodes)
    symxy["displacement 3"] = 0.0
    symxz["displacement 2"] = 0.0
    symyz["displacement 1"] = 0.0
    boundary_problem = Problem(Dirichlet, "symmetry boundary conditions", 3, "displacement")
    push!(boundary_problem, symxy, symxz, symyz)

    solver = Solver("solve 3d block")
    push!(solver, elasticity_problem)
    push!(solver, boundary_problem)
    call(solver)

    disp = element1("displacement", [1.0, 1.0, 1.0], 0.0)
    info("displacement at tip: $disp")
    # verified using Code Aster.
    # 2015-12-12-continuum-elasticity/vim c3d_grot_gdep_traction_force.comm
    @test isapprox(disp, [3.17431158889468E-02, 3.17431158889468E-02, -1.38591518927826E-01])
end
