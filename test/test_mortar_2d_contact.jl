# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Testing

mesh = Mesh()
add_node!(mesh, 1, [0.0, 0.0])
add_node!(mesh, 2, [1.0, 0.0])
add_node!(mesh, 3, [1.0, 0.5])
add_node!(mesh, 4, [0.0, 0.5])
gap = 0.1
add_node!(mesh, 5, [0.0, 0.5+gap])
add_node!(mesh, 6, [1.0, 0.5+gap])
add_node!(mesh, 7, [1.0, 1.0+gap])
add_node!(mesh, 8, [0.0, 1.0+gap])
add_element!(mesh, 1, :Quad4, [1, 2, 3, 4])
add_element!(mesh, 2, :Quad4, [5, 6, 7, 8])
add_element!(mesh, 3, :Seg2, [1, 2])
add_element!(mesh, 4, :Seg2, [7, 8])
add_element!(mesh, 5, :Seg2, [4, 3])
add_element!(mesh, 6, :Seg2, [6, 5])
add_element_to_element_set!(mesh, :LOWER, 1)
add_element_to_element_set!(mesh, :UPPER, 2)
add_element_to_element_set!(mesh, :LOWER_BOTTOM, 3)
add_element_to_element_set!(mesh, :UPPER_TOP, 4)
add_element_to_element_set!(mesh, :LOWER_TOP, 5)
add_element_to_element_set!(mesh, :UPPER_BOTTOM, 6)

upper = Problem(Elasticity, "UPPER", 2)
upper.properties.formulation = :plane_stress
upper.elements = create_elements(mesh, "UPPER")
update!(upper.elements, "youngs modulus", 288.0)
update!(upper.elements, "poissons ratio", 1/3)

lower = Problem(Elasticity, "LOWER", 2)
lower.properties.formulation = :plane_stress
lower.elements = create_elements(mesh, "LOWER")
update!(lower.elements, "youngs modulus", 288.0)
update!(lower.elements, "poissons ratio", 1/3)

bc_upper = Problem(Dirichlet, "UPPER_TOP", 2, "displacement")
bc_upper.elements = create_elements(mesh, "UPPER_TOP")
#update!(bc_upper.elements, "displacement 1", -17/90)
#update!(bc_upper.elements, "displacement 1", -17/90)
update!(bc_upper.elements, "displacement 1", -0.2)
update!(bc_upper.elements, "displacement 2", -0.2)

bc_lower = Problem(Dirichlet, "LOWER_BOTTOM", 2, "displacement")
bc_lower.elements = create_elements(mesh, "LOWER_BOTTOM")
update!(bc_lower.elements, "displacement 1", 0.0)
update!(bc_lower.elements, "displacement 2", 0.0)

contact = Problem(Contact2D, "LOWER_TO_UPPER", 2, "displacement")
contact_slave_elements = create_elements(mesh, "LOWER_TOP")
contact_master_elements = create_elements(mesh, "UPPER_BOTTOM")
add_slave_elements!(contact, contact_slave_elements)
add_master_elements!(contact, contact_master_elements)

solver = Solver(Nonlinear)
push!(solver, upper, lower, bc_upper, bc_lower, contact)

solver()

master = first(contact_master_elements)
slave = first(contact_slave_elements)
um = master("displacement", (0.0,), 0.0)
us = slave("displacement", (0.0,), 0.0)
la = slave("lambda", (0.0,), 0.0)
info("um = $um, us = $us, la = $la")
@test isapprox(um, [-0.20, -0.15])
@test isapprox(us, [0.0, -0.05])
@test isapprox(la, [0.0, 30.375])
