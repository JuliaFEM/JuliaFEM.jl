# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM, Test

#= TODO: Fix test.
@testset "test forwarddiff version + volume load." begin
    nodes = Dict{Int64, Node}(
        1 => [0.0, 0.0],
        2 => [10.0, 0.0],
        3 => [10.0, 1.0],
        4 => [0.0, 1.0])
    # constant volume load on nodes
    load = Dict(
        1 => [0.0, -10.0],
        2 => [0.0, -10.0],
        3 => [0.0, -10.0],
        4 => [0.0, -10.0])
    young = Dict(1 => 500.0, 2 => 500.0, 3 => 500.0, 4 => 500.0)
    poisson = Dict(1 => 0.3, 2 => 0.3, 3 => 0.3, 4 => 0.3)
    element = Quad4([1, 2, 3, 4])
    update!(element, "geometry", nodes)
    update!(element, "youngs modulus", young)
    update!(element, "poissons ratio", poisson)
    update!(element, "displacement load", load)
    boundary = Seg2([1, 4])
    update!(boundary, "geometry", nodes)
    update!(boundary, "displacement 1", 0.0)
    update!(boundary, "displacement 2", 0.0)

    body = Problem(Elasticity, "beam", 2)
    body.properties.formulation = :plane_stress
    body.properties.use_forwarddiff = true
    push!(body, element)
    bc = Problem(Dirichlet, "fixed left side", 2, "displacement")
    #bc.properties.formulation = :incremental
    push!(bc, boundary)

    solver = Solver()
    push!(solver, body, bc)
    solver()
    disp = element("displacement", [1.0, 1.0], 0.0)
    @info("displacement at tip: $disp")
    # verified using Code Aster, verification/2015-10-22-plane-stress/cplan_grot_gdep_volume_force.resu
    @test isapprox(disp[2], -8.77303119819776)
end
=#

#= TODO: Fix test
@testset "test that stiffness matrix is same" begin
    nodes = Dict{Int64, Node}(
        1 => [0.0, 0.0],
        2 => [10.0, 0.0],
        3 => [10.0, 1.0],
        4 => [0.0, 1.0])
    displacement = Dict(
        1 => [0.1, 0.2],
        2 => [0.3, 0.4],
        3 => [0.5, 0.6],
        4 => [0.7, 0.8])
    displacement = Dict(
        1 => [0.0, 0.0],
        2 => [0.0, 0.0],
        3 => [0.0, 0.0],
        4 => [0.0, 0.0])
    load = Dict(
        1 => [0.0, -10.0],
        2 => [0.0, -10.0],
        3 => [0.0, -10.0],
        4 => [0.0, -10.0])
    element = Quad4([1, 2, 3, 4])
    update!(element, "geometry", nodes)
    update!(element, "displacement", displacement)
    update!(element, "youngs modulus", 288.0)
    update!(element, "poissons ratio", 1/3)
    update!(element, "displacement load", load)
    body = Problem(Elasticity, "beam", 2)
    body.properties.formulation = :plane_stress
    K1, f1 = assemble(body, element, 0.0, Val{:forwarddiff})
    K2, f2 = assemble(body, element, 0.0, Val{:plane})
    @test isapprox(K1, K2)
    @test isapprox(f1, f2)
end
=#
