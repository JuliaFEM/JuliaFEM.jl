# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM, Test

#=

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

    element = Element(Hex8, [1, 2, 3, 4, 5, 6, 7, 8])
    update!([element], "geometry", nodes)
    update!([element], "youngs modulus", 200e3)
    update!([element], "poissons ratio", 1/3)

    plastic_parameters = Dict{Any, Any}("type" => JuliaFEM.ideal_plasticity!,
                                        "yield_surface" => Val{:von_mises},
                                        "params" => Dict("yield_stress" => 400.0))
    to_integ_points = Dict()
    map(x-> to_integ_points[x] = plastic_parameters, get_connectivity(element))
    update!(element, "plasticity", to_integ_points)

    elasticity_problem = Problem(Elasticity, "solve continuum block", 3)
    elasticity_problem.properties.finite_strain = false
    elasticity_problem.properties.geometric_stiffness = false
    push!(elasticity_problem.properties.store_fields, :plastic_strain)
    push!(elasticity_problem, element)

    bc = Element(Quad4, [1,4,8,5])
    update!([bc], "geometry", nodes)
    bc["displacement 1"] = 0.0
    bc["displacement 2"] = 0.0
    bc["displacement 3"] = 0.0
    boundary_problem = Problem(Dirichlet, "symmetry boundary conditions", 3, "displacement")
    push!(boundary_problem, bc)

    disp = Element(Quad4, [2,3,7,6])
    update!([disp], "geometry", nodes)
    disp["displacement 1"] = 0.002
    boundary_motion = Problem(Dirichlet, "displacement bc", 3, "displacement")
    push!(boundary_motion, disp)

    solver = NonlinearSolver("solve block problem")
    solver.time = 1.0
    push!(solver, elasticity_problem)
    push!(solver, boundary_problem)
    push!(solver, boundary_motion)
    solver()

    disp = element("displacement", [1.0, 1.0, 1.0], 1.0)
    @info("displacement at tip: $disp")
    u_expected = 2.0 * [-1/3, -1/3, 1.0]
     @test isapprox(disp, u_expected)
end

=#

# function solve_rod_model_elasticity(eltype)
#     fn = @__DIR__() * "/testdata/rod_short.med"
#     mesh = aster_read_mesh(fn, eltype)
#     element_sets = join(keys(mesh.element_sets), ", ")
#     @info("element sets: $element_sets")
#     p1 = Problem(Elasticity, "rod", 3)
#     p2 = Problem(Elasticity, "trac", 3)
#     p3 = Problem(Dirichlet, "fixed", 3, "displacement")
#     p4 = Problem(Dirichlet, "fixed", 3, "displacement")
#     p5 = Problem(Dirichlet, "fixed", 3, "displacement")
#     p1.elements = create_elements(mesh, "ROD")
#     p2.elements = create_elements(mesh, "FACE2")
#     p3.elements = create_elements(mesh, "FACE1")
#     p4.elements = create_elements(mesh, "FACE3")
#     p5.elements = create_elements(mesh, "FACE5")
#     update!(p1, "youngs modulus", 96.0)
#     update!(p1, "poissons ratio", 1/3)
#     update!(p2, "displacement traction force 1", 96.0)
#     update!(p3, "displacement 1", 0.0)
#     update!(p4, "displacement 2", 0.0)
#     update!(p5, "displacement 3", 0.0)
#     solver = LinearSolver(p1, p2, p3, p4, p5)
#     solver()
#     u_max = maximum(p1.assembly.u)
#     @info("$eltype, u_max = $u_max")
#     return u_max
# end
# @testset "compare 3d rod to CA solution" begin
#     @test isapprox(solve_rod_model_elasticity("Tet4"), 0.2)
#     @test isapprox(solve_rod_model_elasticity("Tet10"), 0.2)
#     @test isapprox(solve_rod_model_elasticity("Hex8"), 0.2)
#     @test isapprox(solve_rod_model_elasticity("Hex20"), 0.2)
#     @test isapprox(solve_rod_model_elasticity("Hex27"), 0.2)
# end
