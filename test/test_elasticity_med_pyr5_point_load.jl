# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Testing

@testset "test Pyr5 elasticity with point load" begin
    fn = Pkg.dir("JuliaFEM") * "/geometry/3d_pyr/Pyr5.med"
    mesh = aster_read_mesh(fn)
    element_sets = join(keys(mesh.element_sets), ", ")
    info("element sets: $element_sets")
   
    element1 = create_elements(mesh,"Pyr5")
    baseQuad = create_elements(mesh,"baseQuad")
    tipPoint = Element(Poi1, collect(mesh.node_sets[:tipPoint]))
    
    update!([element1,baseQuad,tipPoint], "geometry", mesh.nodes)
    update!([element1], "youngs modulus", 288.0)
    update!([element1], "poissons ratio", 1/3)
	
    update!([tipPoint], "displacement traction force 1",  5.0)
    update!([tipPoint], "displacement traction force 2", -7.0)
    update!([tipPoint], "displacement traction force 3",  3.0)
	
    elasticity_problem = Problem(Elasticity, "solve continuum block", 3)
    elasticity_problem.properties.finite_strain = false
    push!(elasticity_problem, element1)
    push!(elasticity_problem, tipPoint)
	
    baseQuad[1]["displacement 1"] = 0.0
    baseQuad[1]["displacement 2"] = 0.0
    baseQuad[1]["displacement 3"] = 0.0
    boundary_problem = Problem(Dirichlet, "Boundary conditions", 3, "displacement")
    push!(boundary_problem, baseQuad)
	
    solver = LinearSolver(elasticity_problem, boundary_problem)
    solver()
	
    disp = element1[1]("displacement", [0.0, 0.0, 1.0], 0.0)
    info("########################################################")
    info("displacement at tip: $disp")
    # Code_Aster Result in verification/2017-05-27-pyramids/Pyr5_displacement.txt
    u_expected = [6.9444444444427100E-02,-9.7222222222197952E-02,1.0416666666679683E-02]
    @test isapprox(disp, u_expected)
end
