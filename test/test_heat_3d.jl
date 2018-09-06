# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM, LinearAlgebra, Test

# compare simple 3d heat problem to code aster solution

fn = @__DIR__() * "/testdata/rod_short.med"
mesh = aster_read_mesh(fn, "Hex8")
element_sets = join(keys(mesh.element_sets), ", ")
@debug("element sets: $element_sets")

problem = Problem(Heat, "rod", 1)
volume = create_elements(mesh, "ROD")
face2 = create_elements(mesh, "FACE2")
face3 = create_elements(mesh, "FACE3")
face4 = create_elements(mesh, "FACE4")
face5 = create_elements(mesh, "FACE5")
face6 = create_elements(mesh, "FACE6")
update!(volume, "thermal conductivity", 50.0)
update!(face2, "external temperature", 20.0)
update!(face2, "heat transfer coefficient", 60.0)
update!(face3, "external temperature", 30.0)
update!(face3, "heat transfer coefficient", 50.0)
update!(face4, "external temperature", 40.0)
update!(face4, "heat transfer coefficient", 40.0)
update!(face5, "external temperature", 50.0)
update!(face5, "heat transfer coefficient", 30.0)
update!(face6, "external temperature", 60.0)
update!(face6, "heat transfer coefficient", 20.0)
add_elements!(problem, volume, face2, face3, face4, face5, face6)

bc = Problem(Dirichlet, "left support T=100", 1, "temperature")
bc_elements = create_elements(mesh, "FACE1")
update!(bc_elements, "temperature 1", 100.0)
add_elements!(bc, bc_elements)

analysis = Analysis(Linear)
add_problems!(analysis, problem, bc)
run!(analysis)

# fields extracted from Code Aster .resu file

temp_ca = Dict(
    1 => 1.00000000000000E+02,
    2 => 1.00000000000000E+02,
    3 => 1.00000000000000E+02,
    4 => 1.00000000000000E+02,
    5 => 3.01613322896279E+01,
    6 => 3.01263406641066E+01,
    7 => 3.02559777927923E+01,
    8 => 3.02209215997131E+01)

FLUX_ELGA = Dict(
    1 => [1.74565160615448E+04, -9.99903237329079E+01, -3.69874201221677E+01],
    2 => [1.74565160615448E+04, -3.73168968436642E+02, -1.38038931136833E+02],
    3 => [1.74428571293096E+04, -9.99903237329079E+01, -3.70268090662933E+01],
    4 => [1.74428571293096E+04, -3.73168968436642E+02, -1.38185932677561E+02],
    5 => [1.74615686370955E+04, -9.99509347888079E+01, -3.69874201221677E+01],
    6 => [1.74615686370955E+04, -3.73021966895897E+02, -1.38038931136833E+02],
    7 => [1.74479150854902E+04, -9.99509347888065E+01, -3.70268090662933E+01],
    8 => [1.74479150854901E+04, -3.73021966895874E+02, -1.38185932677561E+02])

FLUX_NOEU = Dict(
    1 => [1.74596669275930E+04,  7.55555618070503E-11,  3.68594044175552E-12],
    2 => [1.74684148339734E+04,  1.10418341137120E-11,  3.48876483258209E-12],
    3 => [1.74360055518019E+04,  7.91828824731056E-11,  1.95399252334028E-13],
    4 => [1.74447696000717E+04, -3.49587025993969E-12,  3.55271367880050E-13],
    5 => [1.74596669275931E+04, -4.73227515822099E+02, -1.74958127606525E+02],
    6 => [1.74684148339733E+04, -4.72904678032251E+02, -1.74958127606524E+02],
    7 => [1.74360055518019E+04, -4.73227515822118E+02, -1.75280965396335E+02],
    8 => [1.74447696000717E+04, -4.72904678032179E+02, -1.75280965396335E+02])

time = 0.0
temp_jf = problem("temperature", time)
@debug("Temperature comparison between JuliaFEM and Code Aster for 3D model",
       temp_ca, temp_jf)

T1 = [temp_ca[i] for i in 1:8]
T2 = [temp_jf[i] for i in 1:8]
@test isapprox(T1, T2; rtol=1.0e-9)
