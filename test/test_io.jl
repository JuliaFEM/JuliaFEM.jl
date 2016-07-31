# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Testing
importall Base

@testset "create new result" begin
    r = ModelIO()
    expected = """<Xdmf xmlns:xi="http://www.w3.org/2001/XInclude" Version="2.1"/>"""
    @test string(r.xdmf) == expected
end

@testset "put and get result" begin
    r = ModelIO()
    put!(r, "/1/2/3", [1 2 3])
    @test isapprox(get(r, "/1/2/3"), [1 2 3])
end

@testset "save results to disk" begin
    X = Dict{Int, Vector{Float64}}(
        1 => [0.0,0.0],
        2 => [1.0,0.0],
        3 => [1.0,1.0],
        4 => [0.0,1.0])
    element = Element(Quad4, [1, 2, 3, 4])
    update!(element, "geometry", X)
    update!(element, "temperature thermal conductivity", 6.0)
    update!(element, "temperature load", 12.0)
    problem = Problem(Heat, "one element heat problem", 1)
    problem.properties.formulation = "2D"
    push!(problem, element)
    boundary_element = Element(Seg2, [1, 2])
    update!(boundary_element, "geometry", X)
    update!(boundary_element, "temperature 1", 0.0)
    bc = Problem(Dirichlet, "fixed", 1, "temperature")
    push!(bc, boundary_element)
    solver = Solver(Linear, problem, bc)
    solver.io = ModelIO()
    solver()
    io = get(solver.io)
    info("h5 file = $(io.name).h5")
    E = get(io, "/Topology/Quad4/Element IDs")
    C = get(io, "/Topology/Quad4/Connectivity")
    N = get(io, "/Node IDs")
    X = get(io, "/Geometry")
    T = get(io, "/Results/Time 0.0/Nodal Fields/Temperature")
    @test isapprox(E, [-1])
    @test isapprox(C, [0 1 2 3])
    @test isapprox(N, [1, 2, 3, 4])
    @test isapprox(X, [0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0]')
    @test isapprox(T, [0.0 0.0 1.0 1.0])
end
