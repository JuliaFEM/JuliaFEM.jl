# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM, Test

meshfile = "/geometry/2d_block/BLOCK_1elem.med"
mesh = aster_read_mesh(dirname(@__DIR__)*meshfile)

# field problem
block = Problem(Elasticity, "BLOCK", 2)
block.properties.formulation = :plane_strain
block.properties.finite_strain = false
block.properties.geometric_stiffness = false

block_elements = create_elements(mesh, "BLOCK")
update!(block_elements, "youngs modulus", 288.0)
update!(block_elements, "poissons ratio", 1/3)
traction_elements = create_elements(mesh, "TOP")
update!(traction_elements, "displacement traction force 2", 288.0*9/8)
add_elements!(block, block_elements, traction_elements)

# boundary conditions
bc = Problem(Dirichlet, "symmetry bc 23", 2, "displacement")
bc_sym_23_elements = create_elements(mesh, "LEFT")
bc_sym_13_elements = create_elements(mesh, "BOTTOM")
update!(bc_sym_23_elements, "displacement 1", 0.0)
update!(bc_sym_13_elements, "displacement 2", 0.0)
add_elements!(bc, bc_sym_23_elements, bc_sym_13_elements)

analysis = Analysis(Linear, block, bc)
run!(analysis)

u = block("displacement", 0.0)
u3_expected = [-0.5, 1.0]
@debug("displacement", u, u3_expected)
@test isapprox(u[3], u3_expected)
