# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using Test

function get_test_model()

    X = Dict{Int, Vector{Float64}}(
         1 => [0.0, 0.0],  2 => [2.0, 0.0],
         3 => [0.0, 1.0],  4 => [2.0, 1.0],
         5 => [0.0, 1.0],  6 => [1.3, 1.0],
         7 => [0.0, 2.0],  8 => [1.3, 2.0],
         9 => [1.3, 1.0], 10 => [2.0, 1.0],
        11 => [1.3, 2.0], 12 => [2.0, 2.0])

    T = Dict{Int, Vector{Float64}}(
         7 => [0.0, 288.0],  8 => [0.0, 288.0],
        11 => [0.0, 288.0], 12 => [0.0, 288.0])

    # volume elements, three bodies
    e1 = Element(Quad4, [1, 2, 4, 3])
    e2 = Element(Quad4, [5, 6, 8, 7])
    e3 = Element(Quad4, [9, 10, 12, 11])
    update!([e1, e2, e3], "geometry", X)
    update!([e1, e2, e3], "youngs modulus", 288.0)
    update!([e1, e2, e3], "poissons ratio", 1/3)
    b1 = Element(Seg2, [7, 8])
    b2 = Element(Seg2, [11, 12])
    update!([b1, b2], "geometry", X)
    update!([b1, b2], "displacement traction force", T)
    body1 = Problem(Elasticity, "body 1", 2)
    body1.properties.formulation = :plane_stress
    push!(body1, e1)
    body2 = Problem(Elasticity, "body 2", 2)
    body2.properties.formulation = :plane_stress
    push!(body2, e2, b1)
    body3 = Problem(Elasticity, "body 3", 2)
    body3.properties.formulation = :plane_stress
    push!(body3, e3, b2)

    # boundary elements for dirichlet dx=0
    dx1 = Element(Seg2, [1, 3])
    dx2 = Element(Seg2, [5, 7])
    update!([dx1, dx2], "geometry", X)
    update!([dx1, dx2], "displacement 1", 0.0)
    bc1 = Problem(Dirichlet, "dx=0", 2, "displacement")
    push!(bc1, dx1, dx2)

    # boundary elements for dirichlet dy=0
    dy1 = Element(Seg2, [1, 2])
    update!(dy1, "geometry", X)
    update!(dy1, "displacement 2", 0.0)
    bc2 = Problem(Dirichlet, "dy=0", 2, "displacement")
    push!(bc2, dy1)

    # mortar boundary between body 1 and body 2
    mel1 = Element(Seg2, [3, 4])
    sel1 = Element(Seg2, [5, 6])
    update!([mel1, sel1], "geometry", X)
    update!(sel1, "master elements", [mel1])
    bc3 = Problem(Mortar, "interface between body 1 and 2", 2, "displacement")
    push!(bc3, mel1, sel1)

    # mortar boundary between body 1 and body 3
    sel2 = Element(Seg2, [9, 10])
    update!(sel2, "geometry", X)
    update!(sel2, "master elements", [mel1])
    bc4 = Problem(Mortar, "interface between body 1 and 3", 2, "displacement")
    push!(bc4, mel1, sel2)

    # mortar boundary between body 2 and body 3
    sel3 = Element(Seg2, [6, 8])
    mel2 = Element(Seg2, [9, 11])
    update!([sel3, mel2], "geometry", X)
    update!(sel3, "master elements", [mel2])
    bc5 = Problem(Mortar, "interface between body 2 and 3", 2, "displacement")
    push!(bc5, sel3, mel2)

    return body1, body2, body3, bc1, bc2, bc3, bc4, bc5
end

#= TODO: Fix test
@testset "test 2d mortar problem with three bodies and shared nodes" begin
    body1, body2, body3, bc1, bc2, bc3, bc4, bc5 = get_test_model()
    solver = Solver(Linear)
    push!(solver, body1, body2, body3, bc1, bc2, bc3, bc4, bc5)
    solver()
    X = e3("geometry", [1.0, 1.0], 0.0)
    u = e3("displacement", [1.0, 1.0], 0.0)
    @info("displacement at $X: $u")
    u_expected = [-1/3, 1.0]
    @test isapprox(u, u_expected)
end
=#
