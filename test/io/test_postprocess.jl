# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM, Test

## interpolate from a set of elements

element1 = Element(Seg2, (1, 2))
element2 = Element(Seg2, (2, 3))
X = Dict(1 => [0.0], 2 => [1.0], 3 => [2.0])
T = Dict(1 => [0.0], 2 => [1.0], 3 => [0.0])
problem = Problem(Heat, "foo", 1)
add_elements!(problem, element1, element2)
problem_elements = get_elements(problem)
update!(problem_elements, "geometry", X)
update!(problem_elements, "temperature", T)
@test isnan(problem("temperature", [-0.1], 0.0))
@test isapprox(problem("temperature", [0.0], 0.0), [0.0])
@test isapprox(problem("temperature", [0.5], 0.0), [0.5])
@test isapprox(problem("temperature", [1.0], 0.0), [1.0])
@test isapprox(problem("temperature", [1.5], 0.0), [0.5])
@test isapprox(problem("temperature", [2.0], 0.0), [0.0])
@test isnan(problem("temperature", [ 2.1], 0.0))
