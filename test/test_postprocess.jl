using JuliaFEM
using JuliaFEM.Testing

@testset "get nodal values" begin
    el1 = Element(Seg2, [1, 2])
    el2 = Element(Seg2, [2, 3])
    X = Dict{Int64, Vector{Float64}}(
        1 => [0.0],
        2 => [1.0],
        3 => [2.0])
    T = Dict{Int64, Vector{Float64}}(
        1 => [0.0],
        2 => [1.0],
        3 => [0.0])
    P = Problem(Heat, "foo", 1)
    push!(P, el1, el2)
    update!(P, "geometry", X)
    update!(P, "temperature", T)
    @test isnan(P("temperature", [-0.1]))
    @test isapprox(P("temperature", [0.0]), [0.0])
    @test isapprox(P("temperature", [0.5]), [0.5])
    @test isapprox(P("temperature", [1.0]), [1.0])
    @test isapprox(P("temperature", [1.5]), [0.5])
    @test isapprox(P("temperature", [2.0]), [0.0])
    @test isnan(P("temperature", [ 2.1]))
end


@testset "interpolate from set of elements" begin
    el1 = Element(Seg2, [1, 2])
    el2 = Element(Seg2, [2, 3])
    X = Dict{Int64, Vector{Float64}}(
        1 => [0.0],
        2 => [1.0],
        3 => [2.0])
    T = Dict{Int64, Vector{Float64}}(
        1 => [0.0],
        2 => [1.0],
        3 => [0.0])
    P = Problem(Heat, "foo", 1)
    push!(P, el1, el2)
    update!(P, "geometry", X)
    update!(P, "temperature", T)
    @test isnan(P("temperature", [-0.1]))
    @test isapprox(P("temperature", [0.0]), [0.0])
    @test isapprox(P("temperature", [0.5]), [0.5])
    @test isapprox(P("temperature", [1.0]), [1.0])
    @test isapprox(P("temperature", [1.5]), [0.5])
    @test isapprox(P("temperature", [2.0]), [0.0])
    @test isnan(P("temperature", [ 2.1]))
end

