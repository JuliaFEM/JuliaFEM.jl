# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Testing

@testset "inverse isoparametric mapping" begin
    el = Element(Quad4, [1, 2, 3, 4])
    X = Dict{Int64, Vector{Float64}}(
        1 => [0.0, 0.0],
        2 => [1.0, 0.0],
        3 => [1.0, 1.0],
        4 => [0.0, 1.0])
    update!(el, "geometry", X)
    time = 0.0
    X1 = el("geometry", [0.1, 0.2], time)
    xi = get_local_coordinates(el, X1, time)
    X2 = el("geometry", xi, time)
    info("X1 = $X1, X2 = $X2")
    @test isapprox(X1, X2)
end

@testset "inside of linear element" begin
    el = Element(Quad4, [1, 2, 3, 4])
    X = Dict{Int64, Vector{Float64}}(
        1 => [0.0, 0.0],
        2 => [1.0, 0.0],
        3 => [1.0, 1.0],
        4 => [0.0, 1.0])
    update!(el, "geometry", X)
    time = 0.0
    @test inside(el, [0.5, 0.5], time) == true
    @test inside(el, [1.0, 0.5], time) == true
    @test inside(el, [1.0, 1.0], time) == true
    @test inside(el, [1.01, 1.0], time) == false
    @test inside(el, [1.0, 1.01], time) == false
end

@testset "inside of quadratic element" begin
    el = Element(Tri6, [1, 2, 3, 4, 5, 6])
    X = Dict{Int64, Vector{Float64}}(
        1 => [0.0, 0.0],
        2 => [1.0, 0.0],
        3 => [0.0, 1.0],
        4 => [0.5, 0.2],
        5 => [0.8, 0.6],
        6 => [-0.2, 0.5])
    update!(el, "geometry", X)
    p = [0.94, 0.3] # visually checked to be inside
    @test inside(el, p, 0.0) == true
    p = [-0.2, 0.8] # visually checked to be outside
    @test inside(el, p, 0.0) == false
end
