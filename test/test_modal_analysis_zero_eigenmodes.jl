# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Postprocess
using JuliaFEM.Testing

@testset "zero eigenmode model" begin
    meshfile = @__DIR__() * "/testdata/primitives.med"
    mesh = aster_read_mesh(meshfile, "TETRA_TET10_1")
    model = Problem(mesh, Elasticity, "TET", 3)
    update!(model.elements, "youngs modulus", 10.0)
    update!(model.elements, "poissons ratio", 1.0/3.0)
    update!(model.elements, "density", 10.0)
    solver = Solver(Modal)
    solver.properties.nev = 30
    push!(solver, model)
    solver()
    freqs = solver.properties.eigvals
    info("freqs: $freqs")
    freqs_expected = [0.0,0.0,0.0,0.0,0.0,0.0,5.05743,5.05743,7.30374,9.55885,9.55885,18.3471,18.3471,19.124,37.008,37.008,47.25,59.8844,63.0,65.3975,65.3975,88.1879,108.854,108.854,131.702,131.702,221.374,301.326,301.326]
    @test isapprox(freqs, freqs_expected; atol=1.0e-3)
end

