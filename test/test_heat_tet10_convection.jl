# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM, LinearAlgebra, Test

mesh_file = @__DIR__() * "/testdata/primitives.med"
mesh = aster_read_mesh(mesh_file, "TETRA_TET10_1")

problem = Problem(Heat, "tet", 1)
problem_elements = create_elements(mesh, "TET")
update!(problem_elements, "thermal conductivity", 50.0)
add_elements!(problem, problem_elements)

face = Problem(Heat, "face 4", 1)
face_elements = create_elements(mesh, "FACE4")
update!(face_elements, "external temperature", 20.0)
update!(face_elements, "heat transfer coefficient", 60.0)
add_elements!(face, face_elements)

fixed = Problem(Dirichlet, "fixed face 3", 1, "temperature")
fixed_elements = create_elements(mesh, "FACE2")
update!(fixed_elements, "temperature 1", 0.0)
@debug("number of elements in fixed set", length(fixed_elements))
add_elements!(fixed, fixed_elements)

analysis = Analysis(Linear)
add_problems!(analysis, problem, face, fixed)
run!(analysis)
Temp = problem("temperature", 0.0)
Temp_expected = Dict(
    1 => 1.45606533688540E+01,
    2 => 0.0,
    3 => 0.0,
    4 => 0.0,
    5 => 1.05228712963739E+01,
    6 => 0.0,
    7 => 9.44202309239159E+00,
    8 => 1.05228712963739E+01,
    9 => 0.0,
    10 => 0.0)

@debug("Temperature solution using JuliaFEM", Temp)
@debug("Temperature solution using Code Aster", Temp_expected)
Temp_diff = collect(Temp[i]-Temp_expected[i] for i in 1:10)
@debug("Difference between solutions", Temp_diff)
@test isapprox(Temp_diff, zeros(10); atol=1.0e-9)
