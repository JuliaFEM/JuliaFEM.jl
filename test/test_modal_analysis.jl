# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Test

@testset "test eigenvalues for single tet4 element" begin
    X = Dict{Int, Vector{Float64}}(
        1 => [2.0, 3.0, 4.0],
        2 => [6.0, 3.0, 2.0],
        3 => [2.0, 5.0, 1.0],
        4 => [4.0, 3.0, 6.0])
    u = Dict{Int, Vector{Float64}}(
        1 => [0.0, 0.0, 0.0],
        2 => [0.0, 0.0, 0.0],
        3 => [0.0, 0.0, 0.0],
        4 => [0.25, 0.25, 0.25])
    e1 = Element(Tet4, [1, 2, 3, 4])
    e2 = Element(Tri3, [1, 2, 3])
    update!([e1, e2], "geometry", X)
    update!([e1, e2], "displacement", u)
    update!(e1, "youngs modulus" => 96.0,
                "poissons ratio" => 1.0/3.0,
                "density" => 420.0)
    update!(e2, "displacement 1" => 0.0,
                "displacement 2" => 0.0,
                "displacement 3" => 0.0)
    p1 = Problem(Elasticity, 3)
    p1.properties.finite_strain = false
    p2 = Problem(Dirichlet, p1)
    push!(p1, e1)
    push!(p2, e2)
    s1 = Solver(Modal)
    s1.properties.which = :LM
    push!(s1, p1, p2)

    call(s1; debug=true)
    @test isapprox(s1.properties.eigvals, [4/3, 1/3])

    p1.properties.geometric_stiffness = true
    s1.properties.geometric_stiffness = true
    call(s1; debug=true)
    @test isapprox(s1.properties.eigvals, [5/3, 2/3])
end
