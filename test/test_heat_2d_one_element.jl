# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM, LinearAlgebra, Test

# 2d heat problem (one element)

X = Dict(
    1 => [0.0,0.0],
    2 => [1.0,0.0],
    3 => [1.0,1.0],
    4 => [0.0,1.0])

# define volume element
element1 = Element(Quad4, (1, 2, 3, 4))

update!(element1, "geometry", X)
update!(element1, "thermal conductivity", 6.0)
update!(element1, "heat source", 12.0)

# define boundary element for flux
element2 = Element(Seg2, (1, 2))
update!(element2, "geometry", X)
# linear ramp from 0 -> 6 in time 0 -> 1
update!(element2, "heat flux", 0.0 => 0.0)
update!(element2, "heat flux", 1.0 => 6.0)

# define heat problem and add elements to problem
problem = Problem(PlaneHeat, "one element heat problem", 1)
add_elements!(problem, element1, element2)

# Set constant source f=12 with k=6. Accurate solution is
# T=1 on free boundary, u(x,y) = -1/6*(1/2*f*x^2 - f*x)
# when boundary flux not active (at t=0)
time = 0.0
assemble!(problem, time)
A = Matrix(problem.assembly.K)
b = Vector(problem.assembly.f)
A_expected = [
     4.0 -1.0 -2.0 -1.0
    -1.0  4.0 -1.0 -2.0
    -2.0 -1.0  4.0 -1.0
    -1.0 -2.0 -1.0  4.0]
free_dofs = [1, 2]
@test isapprox(A, A_expected)
@test isapprox(A[free_dofs, free_dofs] \ b[free_dofs], [1.0, 1.0])

# Set constant flux g=6 on boundary. Accurate solution is
# u(x,y) = x which equals T=1 on boundary.
# at time t=1.0 all loads should be on.
empty!(problem.assembly)
time = 1.0
assemble!(problem, time)
A = Matrix(problem.assembly.K)
b = Vector(problem.assembly.f)
@test isapprox(A[free_dofs, free_dofs] \ b[free_dofs], [2.0, 2.0])
