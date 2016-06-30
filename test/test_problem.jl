# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Test

@testset "test initialize field problem" begin
    el = Element(Seg2, [1, 2])
    pr = Problem(Heat, 1)
    push!(pr, el)
    initialize!(pr)
    @test haskey(el, "temperature")
    # one timestep in field "temperature"
    @test length(el("temperature")) == 1
    @test length(el["temperature"]) == 1
    # length of single increment
    @test length(el("temperature", 0.0)) == 2
    @test length(last(el, "temperature").data) == 2
end

@testset "test initialize boundary problem" begin
    el = Element(Seg2, [1, 2])
    pr = Problem(Dirichlet, "bc", 1, "temperature")
    push!(pr, el)
    initialize!(pr)
    @test haskey(el, "reaction force")
    @test haskey(el, "temperature")
end
