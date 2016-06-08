# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Test

@testset "test tet4 stiffness matrix" begin
    el = Element(Tet4)
    el["youngs modulus"] = 96.0
    el["poissons ratio"] = 1/3
    x1 = [2.0, 3.0, 4.0]
    x2 = [6.0, 3.0, 2.0]
    x3 = [2.0, 5.0, 1.0]
    x4 = [4.0, 3.0, 6.0]
    el["geometry"] = Vector{Float64}[x1, x2, x3, x4]
    pr = Problem(Elasticity, "tet4", 3)
    Kt, f = assemble(pr, el, 0.0, Val{:continuum_linear})
    Kt_expected = [
         149.0   108.0   24.0   -1.0    6.0   12.0  -54.0   -48.0    0.0  -94.0   -66.0  -36.0
         108.0   344.0   54.0  -24.0  104.0   42.0  -24.0  -216.0  -12.0  -60.0  -232.0  -84.0
          24.0    54.0  113.0    0.0   30.0   35.0    0.0   -24.0  -54.0  -24.0   -60.0  -94.0
          -1.0   -24.0    0.0   29.0  -18.0  -12.0  -18.0    24.0    0.0  -10.0    18.0   12.0
           6.0   104.0   30.0  -18.0   44.0   18.0   12.0   -72.0  -12.0    0.0   -76.0  -36.0
          12.0    42.0   35.0  -12.0   18.0   29.0    0.0   -24.0  -18.0    0.0   -36.0  -46.0
         -54.0   -24.0    0.0  -18.0   12.0    0.0   36.0     0.0    0.0   36.0    12.0    0.0
         -48.0  -216.0  -24.0   24.0  -72.0  -24.0    0.0   144.0    0.0   24.0   144.0   48.0
           0.0   -12.0  -54.0    0.0  -12.0  -18.0    0.0     0.0   36.0    0.0    24.0   36.0
         -94.0   -60.0  -24.0  -10.0    0.0    0.0   36.0    24.0    0.0   68.0    36.0   24.0
         -66.0  -232.0  -60.0   18.0  -76.0  -36.0   12.0   144.0   24.0   36.0   164.0   72.0
         -36.0   -84.0  -94.0   12.0  -36.0  -46.0    0.0    48.0   36.0   24.0    72.0  104.0]
    if !isapprox(Kt, Kt_expected)
        info("Test failed")
        info("Kt_expected")
        dump(Kt_expected)
        info("Kt")
        dump(Kt)
    end
    @test isapprox(Kt, Kt_expected)
    Kt, f = assemble(pr, el, 0.0, Val{:continuum})
    @test isapprox(Kt, Kt_expected)
end
