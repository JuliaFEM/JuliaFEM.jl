# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Testing

using JuliaFEM: group_by_element_type

@testset "add time dependent field to element" begin
    el = Element(Seg2, [1, 2])
    u1 = Vector{Float64}[[0.0, 0.0], [0.0, 0.0]]
    u2 = Vector{Float64}[[1.0, 1.0], [1.0, 1.0]]
    update!(el, "displacement", 0.0 => u1)
    update!(el, "displacement", 1.0 => u2)
    @test length(el["displacement"]) == 2
    @test isapprox(el("displacement", [0.0], 0.0), [0.0, 0.0])
    @test isapprox(el("displacement", [0.0], 0.5), [0.5, 0.5])
    @test isapprox(el("displacement", [0.0], 1.0), [1.0, 1.0])
    el2 = Element(Poi1, [1])
    update!(el2, "force 1", 0.0 => 1.0)
end

@testset "add CVTV field to element" begin
    el = Element(Seg2, [1, 2])
    f(xi, time) = xi[1]*time
    update!(el, "my field", f)
    v = el("my field", [1.0], 2.0)
    @test isapprox(v, 2.0)
end

@testset "add DCTI to element" begin
    el = Element(Quad4, [1, 2, 3, 4])
    update!(el, "displacement load", DCTI([4.0, 8.0]))
    @test isa(el["displacement load"], DCTI)
    @test !isa(el["displacement load"].data, DCTI)
    update!(el, "displacement load 2", [4.0, 8.0])
    @test isa(el["displacement load 2"], DCTI)
    update!(el, "temperature", [1.0, 2.0, 3.0, 4.0])
    @test isa(el["temperature"], DVTI)
    @test isapprox(el("displacement load", [0.0, 0.0], 0.0), [4.0, 8.0])
end

@testset "interpolate DCTI from element" begin
    el = Element(Seg2, [1, 2])
    update!(el, "foobar", 1.0)
    fb = el("foobar", [0.0], 0.0)
    @test isa(fb, Float64)
    @test isapprox(fb, 1.0)
end

@testset "add elements to elements" begin
    el1 = Element(Seg2, [1, 2])
    el2 = Element(Seg2, [3, 4])
    update!(el1, "master elements", [el2])
    lst = el1("master elements", 0.0)
    @test isa(lst, Vector)
end

@testset "extend basis" begin
    el = Element(Quad4, [1, 2, 3, 4])
    expected = [
        0.25 0.00 0.25 0.00 0.25 0.00 0.25 0.00
        0.00 0.25 0.00 0.25 0.00 0.25 0.00 0.25]
    @test isapprox(el([0.0, 0.0], 0.0, 2), expected)
end

@testset "group elements" begin
    e1 = Element(Seg2, [1, 2])
    e2 = Element(Quad4, [1, 2, 3, 4])
    elements = [e1, e2]
    r = group_by_element_type(elements)
    @test length(r) == 2
    @test first(r[Element{Seg2}]) == e1
    @test first(r[Element{Quad4}]) == e2
end
