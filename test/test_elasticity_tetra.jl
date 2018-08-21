# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM, Test
using JuliaFEM.Preprocess
using SparseArrays, Test

# test tet4 + volume load

X = Dict(1 => [2.0, 3.0, 4.0], 2 => [6.0, 3.0, 2.0],
         3 => [2.0, 5.0, 1.0], 4 => [4.0, 3.0, 6.0])
element1 = Element(Tet4, (1, 2, 3, 4))
element2 = Element(Tri3, (1, 2, 3))
update!((element1, element2), "geometry", X)
update!(element1, "youngs modulus", 96.0)
update!(element1, "poissons ratio", 1/3)
update!(element1, "displacement load 3", 784.0/110.0)
update!(element2, "displacement 1", 0.0)
update!(element2, "displacement 2", 0.0)
update!(element2, "displacement 3", 0.0)
problem1 = Problem(Elasticity, "tetra", 3)
problem2 = Problem(Dirichlet, "bc", 3, "displacement")
add_element!(problem1, element1)
add_element!(problem2, element2)
analysis = Analysis(Linear)
add_problems!(analysis, problem1, problem2)
run!(analysis)
u4 = problem1("displacement", 0.0)[4]
u4_expected = [-3.0/220.0, -9.0/220.0, 1.0/10.0]
@test isapprox(u4, u4_expected)

# tet4 + surface load

element3 = Element(Tri3, (4, 3, 2))
update!(element3, "geometry", X)
update!(element3, "surface pressure", -96.0)
update!(element1, "displacement load 3", 0.0)
add_element!(problem1, element3)
run!(analysis)
u4 = problem1("displacement", 0.0)[4]
u4_expected = [-17.0/14.0, -27.0/14.0, 1.0]
@debug("displacement at node 4", u4, u4_expected)

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
    @info("la = $la")
    @info("la_expected = $(la_expected)")
    @test isapprox(la, la_expected)
end
=#
