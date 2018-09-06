# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM, LinearAlgebra, Statistics, Test

mesh = aster_read_mesh(@__DIR__()*"/testdata/primitives.med", "CYLINDER_20_TET4")
problem_elements = create_elements(mesh, "CYLINDER")
update!(problem_elements, "thermal conductivity", 200.0)
outer_elements = create_elements(mesh, "FACE2", "FACE3")
update!(outer_elements, "external temperature", 20.0)
update!(outer_elements, "heat transfer coefficient", 1.0)
#midline = Problem(Heat, "midline of rod", 1)
#midline.elements = create_elements(mesh, "INNER_LINE")
boundary_elements = create_elements(mesh, "FACE1")
update!(boundary_elements, "temperature 1", 100.0)

problem = Problem(Heat, "rod of length 20", 1)
add_elements!(problem, problem_elements)
outer = Problem(Heat, "outer surface", 1)
add_elements!(outer, outer_elements)
boundary = Problem(Dirichlet, "homogeneous dirichlet boundary", 1, "temperature")
add_elements!(boundary, boundary_elements)

analysis = Analysis(Linear)
add_problems!(analysis, problem, outer, boundary)
run!(analysis)

# Analytical solution
L = 20
k = 200.0
Tu = 20.0
h = 1.0
P = 2*pi
A = pi
α = h
β = sqrt((h*P)/(k*A))
T0 = 100.0
C = [1.0 1.0; (α+k*β)*exp(β*L) (α-k*β)*exp(-β*L)] \ [T0-Tu, 0]

T_diff = []
for x in range(0, stop=20)
    T_FEM = problem("temperature", [x, 0.0, 0.0], 0.0)[1]
    T_ACC = dot(C, [exp(β*x), exp(-β*x)]) + Tu
    push!(T_diff, norm(T_FEM - T_ACC))
end
@debug("mean diff on heat problem", mean(T_diff))

@test mean(T_diff) < 1.2 # mean diff = 1.14
