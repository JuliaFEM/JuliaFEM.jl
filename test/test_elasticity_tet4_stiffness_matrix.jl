# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM, Test

# test tet4 stiffness matrix

X = Dict(1 => [2.0, 3.0, 4.0],
         2 => [6.0, 3.0, 2.0],
         3 => [2.0, 5.0, 1.0],
         4 => [4.0, 3.0, 6.0])
element = Element(Tet4, (1, 2, 3, 4))
update!(element, "youngs modulus", 96.0)
update!(element, "poissons ratio", 1/3)
update!(element, "geometry", X)
problem = Problem(Elasticity, "tet4", 3)
add_element!(problem, element)
time = 0.0
assemble!(problem, time)
K_expected = [
     149   108   24   -1    6   12  -54   -48    0  -94   -66  -36
     108   344   54  -24  104   42  -24  -216  -12  -60  -232  -84
      24    54  113    0   30   35    0   -24  -54  -24   -60  -94
      -1   -24    0   29  -18  -12  -18    24    0  -10    18   12
       6   104   30  -18   44   18   12   -72  -12    0   -76  -36
      12    42   35  -12   18   29    0   -24  -18    0   -36  -46
     -54   -24    0  -18   12    0   36     0    0   36    12    0
     -48  -216  -24   24  -72  -24    0   144    0   24   144   48
       0   -12  -54    0  -12  -18    0     0   36    0    24   36
     -94   -60  -24  -10    0    0   36    24    0   68    36   24
     -66  -232  -60   18  -76  -36   12   144   24   36   164   72
     -36   -84  -94   12  -36  -46    0    48   36   24    72  104]
@test isapprox(problem.assembly.K, K_expected)
