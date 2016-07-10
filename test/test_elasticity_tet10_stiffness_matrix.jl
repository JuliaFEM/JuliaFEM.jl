# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Testing

@testset "test tet10 stiffness matrix" begin
    el = Element(Tet10, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    el["youngs modulus"] = 480.0
    el["poissons ratio"] = 1/3
    x1 = [2.0, 3.0, 4.0]
    x2 = [6.0, 3.0, 2.0]
    x3 = [2.0, 5.0, 1.0]
    x4 = [4.0, 3.0, 6.0]
    x5 = 0.5*(x1+x2)
    x6 = 0.5*(x2+x3)
    x7 = 0.5*(x3+x1)
    x8 = 0.5*(x1+x4)
    x9 = 0.5*(x2+x4)
    x10 = 0.5*(x3+x4)
    X = Dict{Int64, Vector{Float64}}(
        1 => x1, 2 => x2, 3 => x3, 4 => x4, 5 => x5,
        6 => x6, 7 => x7, 8 => x8, 9 => x9, 10 => x10)
    u = Dict{Int64, Vector{Float64}}()
    for i=1:10
        u[i] = [0.0, 0.0, 0.0]
    end
    update!(el, "geometry", X)
    update!(el, "displacement", u)
    pr = Problem(Elasticity, "tet10", 3)
    ass = Assembly()
    assemble!(ass, pr, el, 0.0)
    Kt = full(ass.K)
    eigs = real(eigvals(Kt))
    eigs_expected = [8809.45, 4936.01, 2880.56, 2491.66, 2004.85,
                     1632.49, 1264.32, 1212.42, 817.905,
                     745.755, 651.034, 517.441, 255.1, 210.955,
                     195.832, 104.008, 72.7562, 64.4376, 53.8515,
                     23.8417, 16.6354, 9.54682, 6.93361, 2.22099,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    @test isapprox(eigs, eigs_expected; atol=1.0e-2)
end
