# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Testing

@testset "test initialize scalar field problem" begin
    el = Element(Seg2, [1, 2])
    pr = Problem(Heat, 1)
    push!(pr, el)
    initialize!(pr)
    @test haskey(el, "temperature")
    # one timestep in field "temperature"
    @test length(el["temperature"]) == 1
    # this way we access to field at default time t=0.0, it's different than ^!
    @test length(el("temperature", 0.0)) == 2
    @test length(last(el, "temperature").data) == 2
end

@testset "test initialize vector field problem" begin
    el = Element(Seg2, [1, 2])
    pr = Problem(Elasticity, 2)
    push!(pr, el)
    initialize!(pr)
    @test haskey(el, "displacement")
    @test length(el["displacement"]) == 1
    # this way we access to field at default time t=0.0, it's different than ^!
    @test length(el("displacement", 0.0)) == 2
    @test length(last(el, "displacement").data) == 2
end

@testset "test initialize boundary problem" begin
    el = Element(Seg2, [1, 2])
    pr = Problem(Dirichlet, "bc", 1, "temperature")
    push!(pr, el)
    initialize!(pr)
    @test haskey(el, "reaction force")
    @test haskey(el, "temperature")
end

#=
@testset "dict field depending from problems" begin
    p1 = Problem(Elasticity, "Body 1", 2)
    p2 = Problem(Elasticity, "Body 2", 2)
    X = Dict{Int64, Vector{Float64}}(
        1 => [0.0, 0.0],
        2 => [1.0, 0.0],
        3 => [1.0, 1.0],
        4 => [0.0, 1.0])
    update!([p1, p2], "geometry", 0.0 => X)
    @test isapprox(p1("geometry", 0.0)[1], [0.0, 0.0])
    @test isapprox(p2("geometry", 0.0)[1], [0.0, 0.0])
    p1("geometry", 0.0)[1] = [1.0, 2.0]
    @test isapprox(p2("geometry", 0.0)[1], [1.0, 2.0])
end

@testset "dict field depending from problems" begin
    X = Dict{Int64, Vector{Float64}}(
        1 => [0.0, 0.0],
        2 => [1.0, 0.0],
        3 => [1.0, 1.0],
        4 => [0.0, 1.0],
        5 => [0.0, 2.0],
        6 => [1.0, 2.0],
        7 => [1.0, 3.0],
        8 => [0.0, 3.0])
    p1 = Problem(Elasticity, "Body 1", 2)
    p2 = Problem(Elasticity, "Body 2", 2)
    e1 = Element(Quad4, [1, 2, 3, 4])
    e2 = Element(Quad4, [5, 6, 7, 8])
    push!(p1, e1)
    push!(p2, e2)
    update!(p1, "geometry", 0.0 => X)
    update!(p2, "geometry", 0.0 => X)
    @test isapprox(p1("geometry", 0.0)[1], [0.0, 0.0])
    @test isapprox(p2("geometry", 0.0)[1], [0.0, 0.0])
    p1("geometry", 0.0)[1] = [1.0, 2.0]
    @test isapprox(p2("geometry", 0.0)[1], [1.0, 2.0])
    @test isapprox(e1("geometry", 0.0)[1], [1.0, 2.0])
end

=#
