# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Test

@testset "test tet4 + volume load" begin
    X1 = [2.0, 3.0, 4.0]
    X2 = [6.0, 3.0, 2.0]
    X3 = [2.0, 5.0, 1.0]
    X4 = [4.0, 3.0, 6.0]
    e1 = Element(Tet4, [1, 2, 3, 4])
    e1["geometry"] = Node[X1, X2, X3, X4]
    e1["youngs modulus"] = 96.0
    e1["poissons ratio"] = 1/3
    e1["displacement load 3"] = 784.0/110.0
    e2 = Element(Tri3, [1, 2, 3])
    e2["geometry"] = Node[X2, X1, X3]
    e2["displacement 1"] = 0.0
    e2["displacement 2"] = 0.0
    e2["displacement 3"] = 0.0
    p1 = Problem(Elasticity, "tetra", 3)
    p2 = Problem(Dirichlet, "bc", 3, "displacement")
    push!(p1, e1)
    push!(p2, e2)
    s = Solver(Linear)
    push!(s, p1, p2)
    s()
    u_4 = p1.assembly.u[10:end]
    u_expected = [-3.0/220.0, -9.0/220.0, 1.0/10.0]
    @test isapprox(u_4, u_expected)
end

@testset "test tet4 + surface load" begin
    nodes = Dict{Int, Vector{Float64}}(
        1 => [2.0, 3.0, 4.0],
        2 => [6.0, 3.0, 2.0],
        3 => [2.0, 5.0, 1.0],
        4 => [4.0, 3.0, 6.0])
    e1 = Element(Tet4, [1, 2, 3, 4])
    update!(e1, "geometry", nodes)
    e1["youngs modulus"] = 96.0
    e1["poissons ratio"] = 1/3
    e2 = Element(Tri3, [1, 2, 3])
    update!(e2, "geometry", nodes)
    e2["displacement 1"] = 0.0
    e2["displacement 2"] = 0.0
    e2["displacement 3"] = 0.0
    e3 = Element(Tri3, [4, 3, 2])
    update!(e3, "geometry", nodes)
    e3["displacement traction force n"] = 96.0
    p1 = Problem(Elasticity, "tetra", 3)
    p2 = Problem(Dirichlet, "bc", 3, "displacement")
    push!(p1, e1, e3)
    push!(p2, e2)
    s = Solver(Linear)
    push!(s, p1, p2)
    s()
    u_4 = p1.assembly.u[10:end]
    u_expected = [-17.0/14.0, -27.0/14.0, 1.0]
    info("u_4 = $(u_4)")
    info("u_expected = $(u_expected)")
    @test isapprox(u_4, u_expected)
end

#= TODO: Fix test. Make linear perturbation solver.
@testset "test tet4 + buckling" begin
    X = Dict{Int, Vector{Float64}}(
        1 => [2.0, 3.0, 4.0],
        2 => [6.0, 3.0, 2.0],
        3 => [2.0, 5.0, 1.0],
        4 => [4.0, 3.0, 6.0])
    u = Dict{Int, Vector{Float64}}(
        1 => [0.0, 0.0, 0.0],
        2 => [0.0, 0.0, 0.0],
        3 => [0.0, 0.0, 0.0],
        4 => [-0.25, -0.25, -0.25])
    e1 = Element(Tet4, [1, 2, 3, 4])
    update!(e1, "geometry", X)
    update!(e1, "displacement", u)
    update!(e1, "youngs modulus", 96.0)
    update!(e1, "poissons ratio", 1/3)
    p1 = Problem(Elasticity, "tetra", 3)
    p1.properties.formulation = :continuum_buckling
    push!(p1, e1)
    assemble!(p1, 0.0)
    free_dofs = [10, 11, 12]
    Km = sparse(p1.assembly.K)[free_dofs, free_dofs]
    Kg = sparse(p1.assembly.Kg)[free_dofs, free_dofs]
    dump(full(Km))
    dump(full(Kg))
    la = sort(eigs(Km, -Kg)[1])
    la_expected = [1.0, 4.0]
    info("la = $la")
    info("la_expected = $(la_expected)")
    @test isapprox(la, la_expected)
end
=#
