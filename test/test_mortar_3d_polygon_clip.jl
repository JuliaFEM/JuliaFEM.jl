# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Testing

@testset "polygon clip case 1" begin
    S = Vector[
        [0.375, 0.0, 0.5],
        [0.6, 0.0, 0.5],
        [0.5, 0.25, 0.5]]
    M = Vector[
        [0.50, 0.0, 0.5],
        [0.25, 0.0, 0.5],
        [0.375, 0.25, 0.5]]
    n0 = [0.0, 0.0, 1.0]
    P = get_polygon_clip(S, M, n0)
    P_expected = Vector{Float64}[
        [0.500, 0.0, 0.5],
        [0.375, 0.0, 0.5],
        [0.4375, 0.125, 0.5]]
    @test length(P) == length(P_expected)
    for (Pi, Pj) in zip(P, P_expected)
        @test isapprox(Pi, Pj)
    end
end

@testset "polygon clip case 2" begin
    S = Vector[
        [0.25, 0.0, 0.5],
        [0.75, 0.0, 0.5],
        [0.50, 0.25, 0.5]]
    M = Vector[
        [0.50, 0.0, 0.5],
        [0.25, 0.0, 0.5],
        [0.375, 0.25, 0.5]]
    n0 = [0.0, 0.0, 1.0]
    P = get_polygon_clip(S, M, n0)
    @test length(P) == 3
end

