# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Test

@testset "test static condensation" begin

    K = sparse([
      4.0  -1.0  -2.0  -1.0
     -1.0   4.0  -1.0  -2.0
     -2.0  -1.0   4.0  -1.0
     -1.0  -2.0  -1.0   4.0])

    f = sparse([6.0, 6.0, 3.0, 3.0])

    I = [1, 2]
    B = [3, 4]

    Kc, fc = eliminate_interior_dofs(K, f, B, I)
    Kc = full(Kc)
    fc = full(fc)
    dump(Kc)
    dump(fc)

    Kc_expected = [
        0.0 0.0  0.0  0.0
        0.0 0.0  0.0  0.0
        0.0 0.0  2.4 -2.4
        0.0 0.0 -2.4  2.4]
   fc_expected = [0.0, 0.0, 9.0, 9.0]

   # TODO: needs to check numbers
   @test isapprox(Kc, Kc_expected)
   @test isapprox(fc, fc_expected)

   #=
   x = sparse(zeros(4))'
   la = sparse(zeros(4))'
   la[3] = la[4] = 24.0
   reconstruct!(cass, x)
   x = full(x)
   info(la)
   info(x)
   @test isapprox(x[1], 1.0)
   @test isapprox(x[2], 1.0)
   =#
end

