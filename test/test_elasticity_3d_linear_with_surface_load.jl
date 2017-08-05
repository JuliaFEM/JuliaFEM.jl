# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Testing

@testset "test continuum 3d linear elasticity with surface load" begin
    nodes = Dict{Int64, Node}(
    1 => [0.0, 0.0, 0.0],
    2 => [1.0, 0.0, 0.0],
    3 => [1.0, 1.0, 0.0],
    4 => [0.0, 1.0, 0.0],
    5 => [0.0, 0.0, 1.0],
    6 => [1.0, 0.0, 1.0],
    7 => [1.0, 1.0, 1.0],
    8 => [0.0, 1.0, 1.0])

    element1 = Element(Hex8, [1, 2, 3, 4, 5, 6, 7, 8])
    element2 = Element(Quad4, [5, 6, 7, 8])
    update!([element1, element2], "geometry", nodes)
    update!([element1], "youngs modulus", 288.0)
    update!([element1], "poissons ratio", 1/3)
    update!([element2], "displacement traction force 3", 288.0)
    update!([element1], "displacement load 3", 576.0)

    elasticity_problem = Problem(Elasticity, "solve continuum block", 3)
    elasticity_problem.properties.finite_strain = false
    push!(elasticity_problem, element1)
    push!(elasticity_problem, element2)

    symxy = Element(Quad4, [1, 2, 3, 4])
    symxz = Element(Quad4, [1, 2, 6, 5])
    symyz = Element(Quad4, [1, 4, 8, 5])
    update!([symxy, symxz, symyz], "geometry", nodes)
    symyz["displacement 1"] = 0.0
    symxz["displacement 2"] = 0.0
    symxy["displacement 3"] = 0.0
    boundary_problem = Problem(Dirichlet, "symmetry boundary conditions", 3, "displacement")
    push!(boundary_problem, symxy, symxz, symyz)

    solver = LinearSolver(elasticity_problem, boundary_problem)
    solver()

    disp = element1("displacement", [1.0, 1.0, 1.0], 0.0)
    info("displacement at tip: $disp")
    u_expected = 2.0 * [-1/3, -1/3, 1.0]
    @test isapprox(disp, u_expected)
end

function solve_rod_model_elasticity(eltype)
    fn = @__DIR__() * "/testdata/rod_short.med"
    mesh = aster_read_mesh(fn, eltype)
    element_sets = join(keys(mesh.element_sets), ", ")
    info("element sets: $element_sets")
    p1 = Problem(Elasticity, "rod", 3)
    p2 = Problem(Elasticity, "trac", 3)
    p3 = Problem(Dirichlet, "fixed", 3, "displacement")
    p4 = Problem(Dirichlet, "fixed", 3, "displacement")
    p5 = Problem(Dirichlet, "fixed", 3, "displacement")
    p1.elements = create_elements(mesh, "ROD")
    p2.elements = create_elements(mesh, "FACE2")
    p3.elements = create_elements(mesh, "FACE1")
    p4.elements = create_elements(mesh, "FACE3")
    p5.elements = create_elements(mesh, "FACE5")
    update!(p1, "youngs modulus", 96.0)
    update!(p1, "poissons ratio", 1/3)
    update!(p2, "displacement traction force 1", 96.0)
    update!(p3, "displacement 1", 0.0)
    update!(p4, "displacement 2", 0.0)
    update!(p5, "displacement 3", 0.0)
    solver = LinearSolver(p1, p2, p3, p4, p5)
    solver()
    u_max = maximum(p1.assembly.u)
    info("$eltype, u_max = $u_max")
    return u_max
end
@testset "compare 3d rod to CA solution" begin
    @test isapprox(solve_rod_model_elasticity("Tet4"), 0.2)
    @test isapprox(solve_rod_model_elasticity("Tet10"), 0.2)
    @test isapprox(solve_rod_model_elasticity("Hex8"), 0.2)
    @test isapprox(solve_rod_model_elasticity("Hex20"), 0.2)
    @test isapprox(solve_rod_model_elasticity("Hex27"), 0.2)
end

