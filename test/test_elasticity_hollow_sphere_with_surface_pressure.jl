# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Testing

#=
test subjects:
- surface pressure load in curved surface
- verification of elements wedge6 and wedge15

from Code Aster, for linear model
 N38      -8.85861895037377E-01 -3.46944695195361E-18 -3.46944695195361E-18
coords of N38 = (1.0, 0.0, 0.0)
 N49      -4.50934684240566E-01 -4.46908405333021E-01 -5.92329377847994E-01
coords of N49 = (0.521002, 0.515482, 0.68032) # quite middle of surface

Analytical solution
uᵣ(r) = b³p/(2Er²(a³-b³)) * (a³(ν+1) + r³(-4ν+2)), where
a = inner surface radial distance, b = outer surface ...

if a=0.9, b=1.0, ν=1/3, E = 24580 and p = 7317 equation yields
-9/10 for radial displacement

http://mms2.ensmp.fr/emms_paris/plasticite3D/exercices/eSpherePress.pdf
=#
function test_wedge_sphere(model, u_CA, S_CA)
    mesh_file = Pkg.dir("JuliaFEM") * "/test/testdata/primitives.med"
    mesh = aster_read_mesh(mesh_file, model)
    body = Problem(Elasticity, "hollow sphere 1/8 model", 3)
    body.elements = create_elements(mesh, "HOLLOWSPHERE8")
    update!(body, "youngs modulus", 24580.0)
    update!(body, "poissons ratio", 1/3)
    bc = Problem(Dirichlet, "symmetry bc", 3, "displacement")
    el1 = create_elements(mesh, "FACE1")
    update!(el1, "displacement 3", 0.0)
    el2 = create_elements(mesh, "FACE2")
    update!(el2, "displacement 2", 0.0)
    el3 = create_elements(mesh, "FACE3")
    update!(el3, "displacement 1", 0.0)
    bc.elements = [el1; el2; el3]
    lo = Problem(Elasticity, "pressure load", 3)
    lo.elements = create_elements(mesh, "OUTER")
    update!(lo, "surface pressure", 7317.0)
    solver = Solver(Linear, body, bc, lo)
    solver()

    X = lo("geometry")
    u = lo("displacement")
    nids = sort(collect(keys(X)))
    umag = Float64[norm(u[id]) for id in nids]
    um = mean(umag)
    us = std(umag)
    rtol = norm(um - 0.9) / max(norm(um), 0.9) * 100.0
    info("mean umag = $um, std umag = $us, rtol = $rtol")
    @test rtol < 1.5 # percents

    info("Verifying displacement against Code Aster solution.. ")
    for nid in keys(u_CA)
        rtol = norm(u[nid] - u_CA[nid]) / max(norm(u[nid]), norm(u_CA[nid])) * 100.0
        info("Node id $nid, rel diff to CA = $rtol %")
        @test isapprox(u[nid], u_CA[nid])
    end

end

@testset """1/8 hollow sphere with surface load""" begin
    u_CA = Dict()
    u_CA[38] = [-8.85861895037377E-01, -3.46944695195361E-18, -3.46944695195361E-18]
    u_CA[49] = [-4.50934684240566E-01, -4.46908405333021E-01, -5.92329377847994E-01]
    S_CA = Dict()
    S_CA[38] = [-1.94504819940510E+03, -3.47014655438479E+04, -3.40876135857114E+04, 1.70379805227866E+03, 2.16707698388864E+03, 4.32009983342141E-12]
    S_CA[49] = [-2.36067010168718E+04, -2.33820775692753E+04, -1.80606705820104E+04, 8.85783922092890E+03, 1.12122431934777E+04, 1.14035796957343E+04]
    test_wedge_sphere("HOLLOWSPHERE8_WEDGE6", u_CA, S_CA)
    #test_wedge_sphere("HOLLOWSPHERE8_WEDGE15")
end

