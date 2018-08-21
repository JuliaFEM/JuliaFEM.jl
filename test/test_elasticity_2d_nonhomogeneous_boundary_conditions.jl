# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM, Test

# Example of 2d nonlinear elasticity with non-homogeneous boundary contitions

# Geometry of nodes
X = Dict(1 => [0.0, 0.0], 2 => [1.0, 0.0], 3 => [1.0, 1.0], 4 => [0.0, 1.0])

block = Problem(Elasticity, "block", 2)
block.properties.formulation = :plane_stress
block.properties.finite_strain = true
block.properties.geometric_stiffness = true

element = Element(Quad4, (1, 2, 3, 4))
update!(element, "geometry", X)
update!(element, "youngs modulus", 288.0)
update!(element, "poissons ratio", 1/3)
add_element!(block, element)

# boundary conditions
bc = Problem(Dirichlet, "bc", 2, "displacement")
bel1 = Element(Seg2, (1, 2))
bel2 = Element(Seg2, (3, 4))
bel3 = Element(Seg2, (4, 1))
update!((bel1, bel2, bel3), "geometry", X)
update!(bel1, "displacement 2", 0.0)
update!(bel2, "displacement 2", 0.5)
update!(bel3, "displacement 1", 0.0)
add_elements!(bc, bel1, bel2, bel3)

analysis = Analysis(Nonlinear)
add_problems!(analysis, block, bc)
run!(analysis)

# results are verified using Code Aster
eps_expected = [-2.08333312468287E-01, 6.25000000000000E-01, 0.0]
sig_expected = [ 4.50685020821470E-06, 4.62857140373777E+02, 0.0]
u3_expected =  [-2.36237356855269E-01, 5.00000000000000E-01]

u3 = block("displacement", 0.0)[3]
@test isapprox(u3, u3_expected, atol=1.0e-5)
