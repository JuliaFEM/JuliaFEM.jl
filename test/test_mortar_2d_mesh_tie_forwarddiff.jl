# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Postprocess
using JuliaFEM.Testing

X = Dict(
    1 => [0.0, 0.0],
    2 => [1.0, 0.0],
    3 => [0.0, 1.0],
    4 => [1.0, 1.0])

u = Dict(
    1 => [0.0, 0.0],
    2 => [0.0, 0.0],
    3 => [0.0, 0.0],
    4 => [0.0, 0.0])

slave = Element(Seg2, [1, 2])
master = Element(Seg2, [3, 4])
update!([slave, master], "geometry", X)
update!([slave, master], "displacement", u)
problem = Problem(Mortar2DAD, "test problem", 2, "displacement")
add_slave_elements!(problem, [slave])
add_master_elements!(problem, [master])
problem.assembly.u = zeros(8)
problem.assembly.la = zeros(8)
assemble!(problem, 0.0)
C1_expected = 1/6*[
               2.0 0.0 1.0 0.0 -2.0 0.0 -1.0 0.0
               0.0 2.0 0.0 1.0 0.0 -2.0 0.0 -1.0
               1.0 0.0 2.0 0.0 -1.0 0.0 -2.0 0.0
               0.0 1.0 0.0 2.0 0.0 -1.0 0.0 -2.0]
C1 = full(problem.assembly.C1)
C2 = full(problem.assembly.C2)
@test isapprox(C1, C2)
@test isapprox(C1, C1_expected)
