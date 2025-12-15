# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM, Test

# zero eigenmode model
X = Dict(
         1 => [2.0, 3.0, 4.0],
         2 => [6.0, 3.0, 2.0],
         3 => [2.0, 5.0, 1.0],
         4 => [4.0, 3.0, 6.0])
X[5] = 1/2*(X[1] + X[2])
X[6] = 1/2*(X[2] + X[3])
X[7] = 1/2*(X[3] + X[1])
X[8] = 1/2*(X[1] + X[4])
X[9] = 1/2*(X[2] + X[4])
X[10] = 1/2*(X[3] + X[4])

element = Element(Tet10, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
update!(element, "youngs modulus", 480.0)
update!(element, "poissons ratio", 1/3)
update!(element, "geometry", X)
update!(element, "density", 105.0)

body = Problem(Elasticity, "TET", 3)
add_elements!(body, element)
analysis = Analysis(Modal)
add_problems!(analysis, body)
analysis.properties.nev = 30
run!(analysis)
w2 = analysis.properties.eigvals
w2_expected = [5.66054e-16, 1.08925e-15, 1.95035e-15, -2.18372e-15, -8.01069e-15, 9.26902e-15, 0.401407, 0.88248, 1.3185, 2.55833, 3.3538, 5.02371, 7.71933, 8.43733, 11.0521, 19.3262, 26.0944, 28.7033, 51.5814, 59.1422, 83.3597, 91.1507, 121.689, 126.753, 157.938, 160.344, 205.634, 324.608, 468.331]
@test isapprox(w2, w2; atol=1.0e-4)
