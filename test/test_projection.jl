# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Testing

@testset "test projection" begin
    C = [
        2.0  1.0  0.0  0.0   0.0   0.0  0.0  0.0
        1.0  2.0  0.0  0.0   0.0   0.0  0.0  0.0
        0.0  0.0  2.0  1.0  -1.0  -2.0  0.0  0.0
        0.0  0.0  1.0  2.0  -2.0  -1.0  0.0  0.0
        0.0  0.0  0.0  0.0   0.0   0.0  0.0  0.0
        0.0  0.0  0.0  0.0   0.0   0.0  0.0  0.0
        0.0  0.0  0.0  0.0   0.0   0.0  0.0  0.0
        0.0  0.0  0.0  0.0   0.0   0.0  0.0  0.0]
    g = [3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    P, h = create_projection(sparse(C), g)
    P_expected = [
        0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
        0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
        0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0
        0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0
        0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0
        0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0
        0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0
        0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0]
    h_expected = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    @test isapprox(full(P), P_expected)
    @test isapprox(full(h), h_expected)
end

