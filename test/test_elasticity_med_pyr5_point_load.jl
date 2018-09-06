# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM, Test

# test Pyr5 elasticity with point load

fn = dirname(@__DIR__) * "/geometry/3d_pyr/Pyr5.med"
mesh = aster_read_mesh(fn)
element_sets = join(keys(mesh.element_sets), ", ")
@debug("element sets: $element_sets")

element1 = create_elements(mesh,"Pyr5")
baseQuad = create_elements(mesh,"baseQuad")
update!(element1, "youngs modulus", 288.0)
update!(element1, "poissons ratio", 1/3)

tip_node_id = first(mesh.node_sets[:tipPoint])
tipPoint = Element(Poi1, [tip_node_id])
update!(tipPoint, "geometry", mesh.nodes)
update!(tipPoint, "displacement traction force 1",  5.0)
update!(tipPoint, "displacement traction force 2", -7.0)
update!(tipPoint, "displacement traction force 3",  3.0)

problem = Problem(Elasticity, "solve continuum block", 3)
problem.properties.finite_strain = false
add_elements!(problem, element1, tipPoint)

update!(first(baseQuad), "displacement 1", 0.0)
update!(first(baseQuad), "displacement 2", 0.0)
update!(first(baseQuad), "displacement 3", 0.0)
bc = Problem(Dirichlet, "Boundary conditions", 3, "displacement")
add_elements!(bc, baseQuad)

analysis = Analysis(Linear)
add_problems!(analysis, problem, bc)
run!(analysis)

xi, time = (0.0, 0.0, 1.0), 0.0
u = first(element1)("displacement", xi, time)
# Code_Aster Result in verification/2017-05-27-pyramids/Pyr5_displacement.txt
u_expected = [6.9444444444427100E-02,-9.7222222222197952E-02,1.0416666666679683E-02]
@debug("displacement at tip", u, u_expected)
@test isapprox(u, u_expected)
