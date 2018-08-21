# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM, Test

# Example of 2d linear elasticity + volume load + surface load

# Dictionary containing node coordinates
X = Dict(1 => [0.0, 0.0], 2 => [1.0, 0.0], 3 => [1.0, 1.0], 4 => [0.0, 1.0])

# Create new problem of type `Elasticity`
block = Problem(Elasticity, "block", 2)
block.properties.formulation = :plane_stress
block.properties.finite_strain = false
block.properties.geometric_stiffness = false

# Add volume element
element = Element(Quad4, (1, 2, 3, 4))
update!(element, "geometry", X)
update!(element, "youngs modulus", 288.0)
update!(element, "poissons ratio", 1/3)
update!(element, "displacement load 2", 576.0)
add_element!(block, element)

# Add boundary element for tractoin force
traction_element = Element(Seg2, (3, 4))
update!(traction_element, "geometry", X)
update!(traction_element, "displacement traction force 2", 288.0)
add_element!(block, traction_element)

# Define boundary conditions
bc = Problem(Dirichlet, "symmetry boundary conditions", 2, "displacement")
bc_elements = [Element(Seg2, (1, 2)), Element(Seg2, (4, 1))]
update!(bc_elements, "geometry", X)
update!(bc_elements[1], "displacement 2", 0.0)
update!(bc_elements[2], "displacement 1", 0.0)
add_elements!(bc, bc_elements)

# Create analysis, add problems to it and run analysis
analysis = Analysis(Linear)
add_problems!(analysis, block, bc)
run!(analysis)

# Analytical solution is known:
f = 288.0
g = 576.0
E = 288.0
nu = 1/3
u3 = block("displacement", 0.0)[3]
u3_expected = f/E*[-nu, 1] + g/(2*E)*[-nu, 1]
@test isapprox(u3, u3_expected)
