# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM, Test

# test continuum nonlinear elasticity with surface load

X = Dict(
    1 => [0.0, 0.0, 0.0],
    2 => [1.0, 0.0, 0.0],
    3 => [1.0, 1.0, 0.0],
    4 => [0.0, 1.0, 0.0],
    5 => [0.0, 0.0, 1.0],
    6 => [1.0, 0.0, 1.0],
    7 => [1.0, 1.0, 1.0],
    8 => [0.0, 1.0, 1.0])

element1 = Element(Hex8, (1, 2, 3, 4, 5, 6, 7, 8))
element2 = Element(Quad4, (5, 6, 7, 8))
update!((element1, element2), "geometry", X)
update!(element1, "youngs modulus", 900.0)
update!(element1, "poissons ratio", 0.25)
update!(element2, "displacement traction force", [0.0, 0.0, -100.0])

problem = Problem(Elasticity, "solve continuum block", 3)
problem.properties.finite_strain = true
add_elements!(problem, element1, element2)

symxy = Element(Quad4, (1, 2, 3, 4))
symxz = Element(Quad4, (1, 2, 6, 5))
symyz = Element(Quad4, (1, 4, 8, 5))
update!((symxy, symxz, symyz), "geometry", X)
update!(symxy, "displacement 3", 0.0)
update!(symxz, "displacement 2", 0.0)
update!(symyz, "displacement 1", 0.0)

bc = Problem(Dirichlet, "symmetry boundary conditions", 3, "displacement")
add_elements!(bc, symxy, symxz, symyz)

analysis = Analysis(Nonlinear)
add_problems!(analysis, problem, bc)
run!(analysis)

u = element1("displacement", [1.0, 1.0, 1.0], 0.0)
u_expected = [3.17431158889468E-02, 3.17431158889468E-02, -1.38591518927826E-01]
@debug("displacement at tip", u, u_expected)

# verified using Code Aster:
# 2015-12-12-continuum-elasticity/vim c3d_grot_gdep_traction_force.comm
@test isapprox(u, u_expected; rtol=1.0e-4)
