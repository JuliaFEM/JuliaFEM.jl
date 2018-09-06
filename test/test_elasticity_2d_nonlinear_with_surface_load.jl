# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM, Test

# Example, 2d nonlinear elasticity with surface load (mesh is read from file)

meshfile = joinpath("test_elasticity_2d_nonlinear_with_surface_load", "BLOCK_1elem.med")
mesh = aster_read_mesh(meshfile)

# define field problem
block = Problem(Elasticity, "BLOCK", 2)
block.properties.formulation = :plane_stress
block.properties.finite_strain = true
block.properties.geometric_stiffness = true

# Add volume elements
block_elements = create_elements(mesh, "BLOCK")
update!(block_elements, "youngs modulus", 288.0)
update!(block_elements, "poissons ratio", 1/3)
update!(block_elements, "displacement load 2", 576.0)
add_elements!(block, block_elements)

# Add surface elements
traction_elements = create_elements(mesh, "TOP")
update!(traction_elements, "displacement traction force 2", 288.0)
add_elements!(block, traction_elements)

# Create boundary problem, add boundary conditions:
bc_sym = Problem(Dirichlet, "symmetry bc", 2, "displacement")
bc_elements_left = create_elements(mesh, "LEFT")
bc_elements_bottom = create_elements(mesh, "BOTTOM")
update!(bc_elements_left, "displacement 1", 0.0)
update!(bc_elements_bottom, "displacement 2", 0.0)
add_elements!(bc_sym, bc_elements_left)
add_elements!(bc_sym, bc_elements_bottom)

# Create analysis, add problems to analysis and run analysis:
analysis = Analysis(Nonlinear)
add_problems!(analysis, block, bc_sym)
run!(analysis)

# Results from Code Aster:
u3_expected = [-4.92316106779943E-01, 7.96321884292103E-01]
eps_zz = -3.71128811855451E-01
eps_expected = [-3.71128532282463E-01, 1.11338615599337E+00, 0.0]
sig_expected = [ 3.36174888827909E-05, 2.23478729403118E+03, 0.0]

u3 = block("displacement", 0.0)[3]
@debug("displacement", u3)
@test isapprox(u3, u3_expected, atol=1.0e-5)
