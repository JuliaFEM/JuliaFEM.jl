# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM, Test, LinearAlgebra

x1 = [2.0, 3.0, 4.0]
x2 = [6.0, 3.0, 2.0]
x3 = [2.0, 5.0, 1.0]
x4 = [4.0, 3.0, 6.0]
x5 = 0.5*(x1+x2)
x6 = 0.5*(x2+x3)
x7 = 0.5*(x3+x1)
x8 = 0.5*(x1+x4)
x9 = 0.5*(x2+x4)
x10 = 0.5*(x3+x4)
X = Dict(
    1 => x1, 2 => x2, 3 => x3, 4 => x4, 5 => x5,
    6 => x6, 7 => x7, 8 => x8, 9 => x9, 10 => x10)
u = Dict(i => zeros(3) for i in 1:10)

element = Element(Tet10, (1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
update!(element, "youngs modulus", 480.0)
update!(element, "poissons ratio", 1/3)
update!(element, "geometry", X)
update!(element, "displacement", u)
problem = Problem(Elasticity, "tet10", 3)
add_element!(problem, element)
time = 0.0
assemble!(problem, time)
eigs = real(eigvals(Matrix(problem.assembly.K)))
eigs_expected = [8809.45, 4936.01, 2880.56, 2491.66, 2004.85,
                 1632.49, 1264.32, 1212.42, 817.905,
                 745.755, 651.034, 517.441, 255.1, 210.955,
                 195.832, 104.008, 72.7562, 64.4376, 53.8515,
                 23.8417, 16.6354, 9.54682, 6.93361, 2.22099,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
@test isapprox(eigs, eigs_expected; atol=1.0e-2)
