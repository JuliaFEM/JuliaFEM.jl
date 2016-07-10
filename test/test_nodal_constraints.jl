# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Testing

function JuliaFEM.get_model(::Type{Val{Symbol("1x1 plane stress quad4 block")}})
    
    X = Dict{Int, Vector{Float64}}(
        1 => [0.0, 0.0],
        2 => [1.0, 0.0],
        3 => [1.0, 1.0],
        4 => [0.0, 1.0])

    body = Problem(Elasticity, "body", 2)
    body.properties.formulation = :plane_stress
    body.elements = [Element(Quad4, [1, 2, 3, 4])]
    update!(body.elements, "geometry", X)
    update!(body.elements, "youngs modulus", 288.0)
    update!(body.elements, "poissons ratio", 1/3)

    # boundary conditions
    bc_13 = Problem(Dirichlet, "symmetry 13", 2, "displacement")
    bc_13.properties.dual_basis = true
    bc_13.elements = [Element(Seg2, [1, 2])]
    update!(bc_13.elements, "geometry", X)
    update!(bc_13.elements, "displacement 2", 0.0)

    bc_23 = Problem(Dirichlet, "symmetry 23", 2, "displacement")
    bc_23.properties.dual_basis = true
    bc_23.elements = [Element(Seg2, [4, 1])]
    update!(bc_23.elements, "geometry", X)
    update!(bc_23, "displacement 1", 0.0)

    solver = Solver(Nonlinear, "1x1 plane stress quad4 block")
    push!(solver, body, bc_13, bc_23)

    return solver
end

@testset "test dirichlet spc in point" begin
    X = Dict{Int, Vector{Float64}}(
        1 => [0.0, 0.0],
        2 => [1.0, 0.0],
        3 => [1.0, 1.0],
        4 => [0.0, 1.0])
    solver = get_model("1x1 plane stress quad4 block")
    update!(solver["symmetry 13"], "displacement 1", 0.0)
    update!(solver["symmetry 23"], "displacement 2", 0.0)
    nodal_bc = Problem(Dirichlet, "dx=0.5", 2, "displacement")
    nodal_bc.elements = [Element(Poi1, [3])]
    update!(nodal_bc, "geometry", X)
    update!(nodal_bc, "displacement 1", 0.5)
    update!(nodal_bc, "displacement 2", 0.0)
    push!(solver, nodal_bc)
    solver()
    pel = nodal_bc.elements[1]
    la = pel("reaction force", [0.0], 0.0)
    info("reaction force: $la")
    info(solver["body"].assembly.u)
    @test isapprox(pel("displacement", [], 0.0), [0.5, 0.0])
end

@testset "test nodal point force" begin
    X = Dict{Int, Vector{Float64}}(
        1 => [0.0, 0.0],
        2 => [1.0, 0.0],
        3 => [1.0, 1.0],
        4 => [0.0, 1.0])
    solver = get_model("1x1 plane stress quad4 block")
    update!(solver["symmetry 13"], "displacement 1", 0.0)
    update!(solver["symmetry 23"], "displacement 2", 0.0)
    point_load = Element(Poi1, [3])
    update!(point_load, "geometry", X)
    update!(point_load, "displacement traction force 1", 72.0)
    update!(point_load, "displacement traction force 2", 27.0)
    push!(solver["body"], point_load)
    solver()
    @test isapprox(point_load("displacement", [], 0.0), [0.5, 0.0])
end

