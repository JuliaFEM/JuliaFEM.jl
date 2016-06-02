# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Test

@testset "find intersection of Seg3 element" begin
    el = Element(Seg3, [1, 2, 3])
    update!(el, "geometry", Vector{Float64}[[0.0, 1.0], [1.0, 0.0], sqrt(2.0)/2.0*[1.0, 1.0]])
    s = [ 1.0, 0.5]
    d = [-1.0, 0.0]
    t, xi = find_intersection(el, s, d, 0.0)
    x = el("geometry", xi, 0.0)
    @test isapprox(x, [0.8604093371313943, 0.5])
end

@testset "find intersection of 2. order NSeg element" begin
    # this is a exact quarter of circle
    a = 1.0
    b = 1.0
    el = Element(NSeg, [1, 2, 3])
    el.properties.order = 2
    el.properties.knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    el.properties.weights = [1.0, 1.0, 2.0]
    update!(el, "geometry", Vector{Float64}[[a, 0], [a, b], [0, b]])
    s = [ 1.0, 0.5]
    d = [-1.0, 0.0]
    t, xi = find_intersection(el, s, d, 0.0)
    X = el("geometry", xi, 0.0)
    @test isapprox(X, [sqrt(3)/2, 1/2])
end

@testset "find intersection of 1. order NSurf" begin
    # node ordering, it's not same as in Quad4
    el = Element(NSurf, [1, 2, 3, 4])
    el.properties.order_u = 1
    el.properties.order_v = 1
    el.properties.knots_u = [0.0, 0.0, 1.0, 1.0]
    el.properties.knots_v = [0.0, 0.0, 1.0, 1.0]
    el.properties.weights = ones(2, 2)
    # +-- v
    # |
    # u
    nodes = Vector{Float64}[[0.0,0.0,0.0], [1.0,0.0,0.0],
                            [0.0,1.0,0.0], [1.0,1.0,0.0]]
    update!(el, "geometry", nodes)
    s = [0.5, 0.5,  0.5]
    d = [0.0, 0.0, -1.0]
    t, xi = find_intersection(el, s, d, 0.0)
    X = el("geometry", xi, 0.0)
    @test isapprox(X, [0.5, 0.5, 0.0])
    k = calc_reflection(el, xi, d, 0.0)
    @test isapprox(k, [0.0, 0.0, 1.0])
    s = [1.5, 1.5, 0.5]
    t, xi = find_intersection(el, s, d, 0.0)
    @test isnan(t)
end

@testset "find intersection of 3. order NSeg element with multiple reflections" begin
    el = Element(NSeg, [1, 2, 3, 4])
    el.properties.order = 3
    el.properties.knots = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    el.properties.weights = [1.0, 1/3, 1/3, 1.0]
    update!(el, "geometry", Vector{Float64}[[1, 0], [1, 2], [-1, 2], [-1, 0]])
    s1 = [-sqrt(3)/2.0, 0.0]
    k1 = [ 0.0, 1.0]
    t1, xi1 = find_intersection(el, s1, k1, 0.0)
    s2 = el("geometry", xi1, 0.0)
    @test isapprox(s2, [-sqrt(3)/2, 0.5])
    k2 = calc_reflection(el, xi1, k1, 0.0)
    t2, xi2 = find_intersection(el, s2, k2, 0.0; secondary=true)
    s3 = el("geometry", xi2, 0.0)
    @test isapprox(s3, [0.0, 1.0])
    k3 = calc_reflection(el, xi2, k2, 0.0)
    t3, xi3 = find_intersection(el, s3, k3, 0.0; secondary=true)
    s4 = el("geometry", xi3, 0.0)
    @test isapprox(s4, [sqrt(3)/2, 0.5])
end

