# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM, Test

# Local stiffness matrix of plane stress element

element = Element(Quad4, (1, 2, 3, 4))
X = Dict(1 => [0.0, 0.0], 2 => [1.0, 0.0], 3 => [1.0, 1.0], 4 => [0.0, 1.0])
u = Dict(1 => [0.0, 0.0], 2 => [0.0, 0.0], 3 => [0.0, 0.0], 4 => [0.0, 0.0])
update!(element, "geometry", X)
update!(element, "displacement", u)
update!(element, "youngs modulus", 288.0)
update!(element, "poissons ratio", 1/3)
update!(element, "displacement load", [4.0, 8.0])

problem = Problem(Elasticity, "[0x1] x [0x1] block", 2)
problem.properties.formulation  = :plane_stress
add_element!(problem, element)
assemble!(problem, 0.0)

K = Matrix(problem.assembly.K)
f = Vector(problem.assembly.f)

K_expected = [
    144  54 -90   0 -72 -54  18   0
     54 144   0  18 -54 -72   0 -90
    -90   0 144 -54  18   0 -72  54
      0  18 -54 144   0 -90  54 -72
    -72 -54  18   0 144  54 -90   0
    -54 -72   0 -90  54 144   0  18
     18   0 -72  54 -90   0 144 -54
      0 -90  54 -72   0  18 -54 144]

f_expected = [1, 2, 1, 2, 1, 2, 1, 2]

@test isapprox(K, K_expected)
@test isapprox(f, f_expected)
