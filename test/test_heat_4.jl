# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM, Test

X = Dict(1 => [0.0,0.0], 2 => [1.0,0.0], 3 => [1.0,1.0], 4 => [0.0,1.0])
element = Element(Quad4, (1, 2, 3, 4))
update!(element, "geometry", X)
update!(element, "thermal conductivity", 6.0)
update!(element, "heat source", 0.0 => 12.0)
update!(element, "heat source", 1.0 => 24.0)
problem = Problem(PlaneHeat, "one element heat problem", 1)
add_element!(problem, element)
boundary_element = Element(Seg2, [1, 2])
update!(boundary_element, "geometry", X)
update!(boundary_element, "temperature 1", 0.0)
bc = Problem(Dirichlet, "fixed", 1, "temperature")
add_element!(bc, boundary_element)
analysis = Analysis(Linear)
add_problems!(analysis, problem, bc)

# two increments, linear solver

run!(analysis)
@test isapprox(analysis("temperature", 0.0)[3], 1.0)

empty!(problem.assembly)
run!(analysis)
@test isapprox(analysis("temperature", 0.0)[3], 1.0)

empty!(problem.assembly)
analysis.properties.time = 1.0
run!(analysis)
@test isapprox(analysis("temperature", 1.0)[3], 2.0)

empty!(problem.assembly)
run!(analysis)
@test isapprox(analysis("temperature", 1.0)[3], 2.0)

# two increments, nonlinear solver

delete!(element.fields, "temperature")
analysis = Analysis(Nonlinear)
add_problems!(analysis, problem, bc)

empty!(problem.assembly)
run!(analysis)
@test isapprox(analysis("temperature", 0.0)[3], 1.0)

empty!(problem.assembly)
run!(analysis)
@test isapprox(analysis("temperature", 0.0)[3], 1.0)

empty!(problem.assembly)
analysis.properties.time = 1.0
run!(analysis)
@test isapprox(analysis("temperature", 1.0)[3], 2.0)

empty!(problem.assembly)
run!(analysis)
@test isapprox(analysis("temperature", 1.0)[3], 2.0)
