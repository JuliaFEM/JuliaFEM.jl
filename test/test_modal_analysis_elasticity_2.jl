# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM, Test

# eigenvalues of CYLINDER1

meshfile = @__DIR__() * "/testdata/primitives.med"
mesh = aster_read_mesh(meshfile, "CYLINDER_1_TET4")
cylinder = Problem(Elasticity, "CYLINDER", 3)
cylinder_elements = create_elements(mesh, "CYLINDER")
update!(cylinder_elements, "youngs modulus", 10000.0)
update!(cylinder_elements, "poissons ratio", 0.3)
update!(cylinder_elements, "density", 10.0)
add_elements!(cylinder, cylinder_elements)

bc = Problem(Dirichlet, "bc", 3, "displacement")
bc_elements = create_elements(mesh, "FACE_YZ1")
update!(bc_elements, "displacement 1", 0.0)
update!(bc_elements, "displacement 2", 0.0)
update!(bc_elements, "displacement 3", 0.0)
add_elements!(bc, bc_elements)

analysis = Analysis(Modal)
analysis.properties.nev = 3
add_problems!(analysis, cylinder, bc)
run!(analysis)
freqs_jf = sqrt.(analysis.properties.eigvals)/(2.0*pi)
freqs_ca = [4.84532E+00, 4.90698E+00, 8.33813E+00]
@test isapprox(freqs_jf, freqs_ca; rtol=1.0e-5)

# eigenvalues of CYLINDER20

meshfile = @__DIR__() * "/testdata/primitives.med"
mesh = aster_read_mesh(meshfile, "CYLINDER_20_TET4")
cylinder = Problem(Elasticity, "CYLINDER", 3)
cylinder_elements = create_elements(mesh, "CYLINDER")
#update!(cylinder_elements, "youngs modulus", 10.0e6)
update!(cylinder_elements, "youngs modulus", 50475.5)
update!(cylinder_elements, "poissons ratio", 0.3)
#update!(cylinder_elements, "density", 10.0)
update!(cylinder_elements, "density", 1.0)
add_elements!(cylinder, cylinder_elements)


bc = Problem(Dirichlet, "bc", 3, "displacement")
bc_elements = create_elements(mesh, "FACE1", "FACE2")
update!(bc_elements, "displacement 1", 0.0)
update!(bc_elements, "displacement 2", 0.0)
update!(bc_elements, "displacement 3", 0.0)
add_elements!(bc, bc_elements)

analysis = Analysis(Modal)
analysis.properties.nev = 3
add_problems!(analysis, cylinder, bc)
run!(analysis)
freqs_jf = sqrt.(analysis.properties.eigvals)/(2.0*pi)
freqs_ca = [1.19789E+00, 1.20179E+00, 3.07391E+00]
@test isapprox(freqs_jf, freqs_ca; rtol=1.0e-5)
