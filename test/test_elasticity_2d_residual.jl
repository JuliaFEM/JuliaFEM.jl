# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM, Test

# Test stiffness matrix of geometrically nonlinear problem

X = Dict(1 => [0.0, 0.0], 2 => [1.0, 0.0], 3 => [1.0, 1.0], 4 => [0.0, 1.0])
u = Dict(1 => [0.1, 0.2], 2 => [0.3, 0.4], 3 => [0.5, 0.6], 4 => [0.7, 0.8])
T = Dict(3 => [0.0, 288.0], 4 => [0.0, 288.0])

element = Element(Quad4, (1, 2, 3, 4))
update!(element, "geometry", X)
update!(element, "displacement", u)
update!(element, "youngs modulus", 288.0)
update!(element, "poissons ratio", 1/3)

traction = Element(Seg2, (3, 4))
update!(traction, "geometry", X)
update!(traction, "displacement", u)
update!(traction, "displacement traction force", T)

# field problem
block = Problem(Elasticity, "block", 2)
block.properties.formulation = :plane_stress
block.properties.finite_strain = true
block.properties.geometric_stiffness = true
add_element!(block, element)
#add_elements!(block, traction)

assemble!(block, 0.0)
Km = Matrix(block.assembly.K)
Kg = Matrix(block.assembly.Kg)
K = Km + Kg
f = Vector(block.assembly.f)

K_expected = [
  401.76   200.88  -123.12    -5.76  -191.52  -117.36  -87.12   -77.76
  200.88   473.76    -5.76    28.08  -117.36  -205.92  -77.76  -295.92
 -123.12    -5.76   197.28    -2.16   -12.24   -25.92  -61.92    33.84
   -5.76    28.08    -2.16   298.08   -25.92  -163.44   33.84  -162.72
 -191.52  -117.36   -12.24   -25.92   240.48   120.24  -36.72    23.04
 -117.36  -205.92   -25.92  -163.44   120.24   312.48   23.04    56.88
  -87.12   -77.76   -61.92    33.84   -36.72    23.04  185.76    20.88
  -77.76  -295.92    33.84  -162.72    23.04    56.88   20.88   401.76]
#   f_expected = [142.272, 214.272, -13.824, 58.176, -75.456, 19.584, -52.992, -4.032]
f_expected = [142.272, 214.272, -13.824, 58.176, -75.456, -124.416, -52.992, -148.032]
@test isapprox(K, K_expected)
@test isapprox(f, f_expected)
