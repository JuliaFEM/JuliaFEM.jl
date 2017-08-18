# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Postprocess
using JuliaFEM.Testing

@testset "eigenvalues of CYLINDER1" begin
    meshfile = @__DIR__() * "/testdata/primitives.med"
    mesh = aster_read_mesh(meshfile, "CYLINDER_1_TET4")
    cylinder = Problem(mesh, Elasticity, "CYLINDER", 3)
    update!(cylinder.elements, "youngs modulus", 10000.0)
    update!(cylinder.elements, "poissons ratio", 0.3)
    update!(cylinder.elements, "density", 10.0)
    bc1 = create_elements(mesh, "FACE_YZ1")
    update!(bc1, "displacement 1", 0.0)
    update!(bc1, "displacement 2", 0.0)
    update!(bc1, "displacement 3", 0.0)
    bcs = Problem(Dirichlet, "bcs", 3, "displacement")
    bcs.elements = bc1
    solver = Solver(Modal)
    solver.properties.nev = 3
    push!(solver, cylinder, bcs)
    solver()
    freqs_jf = sqrt.(solver.properties.eigvals)/(2.0*pi)
    freqs_ca = [4.84532E+00, 4.90698E+00, 8.33813E+00]
    passed = []
    for (f1, f2) in zip(freqs_jf, freqs_ca)
        rtol = norm(f1-f2) / max(f1,f2)
        @printf "JF: %8.5e | CA: %8.5e | rtol: %8.5e\n" f1 f2 rtol
        push!(passed, rtol < 1.0e-5)
    end
    @test reduce(&, passed)
end

@testset "eigenvalues of CYLINDER20" begin
    meshfile = @__DIR__() * "/testdata/primitives.med"
    mesh = aster_read_mesh(meshfile, "CYLINDER_20_TET4")
    cylinder = Problem(mesh, Elasticity, "CYLINDER", 3)
    #update!(cylinder.elements, "youngs modulus", 10.0e6)
    update!(cylinder.elements, "youngs modulus", 50475.5)
    update!(cylinder.elements, "poissons ratio", 0.3)
    #update!(cylinder.elements, "density", 10.0)
    update!(cylinder.elements, "density", 1.0)
    bc1 = create_elements(mesh, "FACE1", "FACE2")
    update!(bc1, "displacement 1", 0.0)
    update!(bc1, "displacement 2", 0.0)
    update!(bc1, "displacement 3", 0.0)
    bcs = Problem(Dirichlet, "bcs", 3, "displacement")
    bcs.elements = bc1
    solver = Solver(Modal)
    solver.properties.nev = 3
    push!(solver, cylinder, bcs)
    solver()
    freqs_jf = sqrt.(solver.properties.eigvals)/(2.0*pi)
    #freqs_ca = [8.82848E-01, 8.85353E-01, 5.30286E+00] # only face1 fixed
    #freqs_ca = [5.33185E+00, 5.34920E+00, 1.36820E+01] # face1 and face2 fixed
    freqs_ca = [1.19789E+00, 1.20179E+00, 3.07391E+00]
    passed = []
    for (f1, f2) in zip(freqs_jf, freqs_ca)
        rtol = norm(f1-f2) / max(f1,f2)
        @printf "JF: %8.5e | CA: %8.5e | rtol: %8.5e\n" f1 f2 rtol
        push!(passed, rtol < 1.0e-5)
    end
    @test reduce(&, passed)
end

