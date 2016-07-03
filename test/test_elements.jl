# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Test

#=
function test_interpolate()
    el = get_element()
    @test isapprox(el("geometry", [0.0, 0.0]), [0.5, 0.5])
    @test isapprox(el("geometry", [0.0, 0.0], 0.0), [0.5, 0.5])
    @test isapprox(el([0.0, 0.0]), [0.25 0.25 0.25 0.25])
    @test isapprox(el([0.0, 0.0], Val{:grad}), [-0.5 0.5 0.5 -0.5; -0.5 -0.5 0.5 0.5])
    gradT = el("temperature", [0.0, 0.0], 1.0, Val{:grad})
    info("gradT = $gradT")
    X = [0.5, 0.5]
    gradT_expected = [1-2*X[2] 3-2*X[1]]
    info("gradT(expected) = $gradT_expected")
    @test isapprox(gradT, gradT_expected)

#   @test isapprox(el("temperature", [0.0, 0.0], 0.5), 1/2*gradT_expected)

#   gradT = el("temperature", [0.0, 0.0], 0.5, Val{:grad})
#   info("gradT = $gradT")
#   @test isapprox(gradT, 1/2*gradT_expected)
end

function test_calculate_normal_tangential_coordinates()
    el = Tri3([1, 2, 3])
    el["geometry"] = Vector{Float64}[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]]
    calculate_normal_tangential_coordinates!(el, 0.0)
    n = [0.0 0.0 1.0]'
    t1 = [1.0 0.0 0.0]'
    t2 = [0.0 1.0 0.0]'
    R = [n t1 t2]
    @test isapprox(el("normal-tangential coordinates", [0.0, 0.0], 0.0), R)
end

function test_manifold_determinant()
    el = Quad4([1, 2, 3, 4])
    #el["geometry"] = Vector{Float64}[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    el["geometry"] = Vector{Float64}[[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]
    # mother element area = 2*2 = 4, this element is 1, determinant should be 1/4 everywhere
    d = det(el, [0.1, 0.2], 0.0)
    d_expected = 0.25
    @test d == d_expected
end

@testset "add new discrete constant time-variant field and interpolate it" begin
    element = Element(Quad4, [1, 2, 3, 4])
    element["my field"] = (0.0 => 0.0, 1.0 => 1.0)
    @test isapprox(element("my field", [0.0, 0.0], 0.5), 0.5)
    update!(element, "my field 2", 0.0 => 0.0, 1.0 => 1.0)
    @test isapprox(element("my field 2", [0.0, 0.0], 0.5), 0.5)
end

=#

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
end

@testset "interpolate DCTI from element" begin
    el = Element(Seg2, [1, 2])
    update!(el, "foobar", 1.0)
    fb = el("foobar", [0.0], 0.0)
    @test isa(fb, Float64)
    @test isapprox(fb, 1.0)
end

@testset "add two time dependent fields to element at once" begin
    el = Element(Seg2, [1, 2])
    update!(el, "foo1", 1.0 => 1.0)
    update!(el, "foo1", 2.0 => 2.0)
    update!(el, "foo2", 1.0 => 1.0, 2.0 => 2.0)
    @test isapprox(el("foo1", 1.5), el("foo2", 1.5))
end

@testset "add elements to elements" begin
    el1 = Element(Seg2, [1, 2])
    el2 = Element(Seg2, [3, 4])
    update!(el1, "master elements", [el2])
    lst = el1("master elements", 0.0)
    info("lst = ", el1["master elements"])
    info("typeof lst = ", typeof(lst))
    @test isa(lst, Vector)
end

