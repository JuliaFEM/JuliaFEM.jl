# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Testing

#=
test subjects:
- surface pressure load in curved surface
- verification of elements wedge6 and wedge15

from Code Aster:
N38      -8.85861895037377E-01 -3.46944695195361E-18 -3.46944695195361E-18
coords of N38 = (1.0, 0.0, 0.0)

Analytical solution
uᵣ(r) = b³p/(2Er²(a³-b³)) * (a³(ν+1) + r³(-4ν+2)), where
a = inner surface radial distance, b = outer surface ...

if a=0.9, b=1.0, ν=1/3, E = 24580 and p = 7317 equation yields
-9/10 for radial displacement
=#

@testset """1/8 hollow sphere with surface load""" begin
    mesh_file = Pkg.dir("JuliaFEM") * "/test/testdata/primitives.med"
    mesh = aster_read_mesh(mesh_file, "HOLLOWSPHERE8_WEDGE6")
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
    solver = LinearSolver(body, bc, lo)
    solver()

    X = lo("geometry")
    u = lo("displacement")
    nids = sort(collect(keys(X)))
    umag = Float64[norm(u[id]) for id in nids]
    um = mean(umag)
    info("mean umag = $um")
    info("std umag = ", std(umag))
    rtol = norm(um - 0.9) / max(norm(um), 0.9) * 100.0
    info("rtol = $rtol")
    @test rtol < 1.5 # percents
    u_CA = [-8.85861895037377E-01, -3.46944695195361E-18, -3.46944695195361E-18]
    rtol = norm(u[38] - u_CA) / max(norm(u[38]), norm(u_CA)) * 100.0
    info("rel diff to CA = $rtol %")
    @test isapprox(u[38], u_CA)
end
