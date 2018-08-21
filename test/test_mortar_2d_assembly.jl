# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using Test

function get_test_2d_model()
    X = Dict{Int64, Vector{Float64}}(
        1 => [0.0, 1.0],
        2 => [3/4, 1.0],
        3 => [2.0, 1.0],
        4 => [0.0, 1.0],
        5 => [5/4, 1.0],
        6 => [2.0, 1.0])
    sel1 = Element(Seg2, [1, 2])
    sel2 = Element(Seg2, [2, 3])
    mel1 = Element(Seg2, [4, 5])
    mel2 = Element(Seg2, [5, 6])
    update!([mel1, mel2, sel1, sel2], "geometry", X)
    return [sel1, sel2], [mel1, mel2]
end

@testset "calculate flat 2d assembly" begin
    (sel1, sel2), (mel1, mel2) = get_test_2d_model()

    bc = Problem(Mortar2D, "test interface", 1, "temperature")
    add_slave_elements!(bc, [sel1, sel2])
    add_master_elements!(bc, [mel1, mel2])

    B_expected = zeros(3, 6)

    S1 = get_gdofs(bc, sel1)
    M1 = get_gdofs(bc, mel1)
    B_expected[S1,S1] += [1/4 1/8; 1/8 1/4]
    B_expected[S1,M1] -= [3/10 3/40; 9/40 3/20]

    S2 = get_gdofs(bc, sel2)
    M2 = get_gdofs(bc, mel1)
    B_expected[S2,S2] += [49/150 11/150; 11/150 2/75]
    B_expected[S2,M2] -= [13/150 47/150; 1/75 13/150]

    S3 = get_gdofs(bc, sel2)
    M3 = get_gdofs(bc, mel2)
    B_expected[S3,S3] += [9/100 27/200; 27/200 39/100]
    B_expected[S3,M3] -= [3/20 3/40; 9/40 3/10]

    assemble!(bc, 0.0)
    B = full(bc.assembly.C1, 3, 6)
    # dump(round(B, 6))
    # dump(B_expected)
    @test isapprox(B, B_expected; rtol=1.0e-9)

end

@testset "solve mortar tie contact with multiple dirichlet boundary conditions and multiple bodies" begin
    X = Dict{Int64, Vector{Float64}}(
        1 => [0.0, 0.0], 2 => [1.0, 0.0],
        3 => [0.0, 0.5], 4 => [1.0, 0.5],
        5 => [0.0, 0.5], 6 => [1.0, 0.5],
        7 => [0.0, 1.0], 8 => [1.0, 1.0])
    T = Dict{Int64, Vector{Float64}}(
        7 => [0.0, 288.0], 8 => [0.0, 288.0]
    )
    e1 = Element(Quad4, [1, 2, 4, 3])
    e2 = Element(Quad4, [5, 6, 8, 7])
    t1 = Element(Seg2, [7, 8])
    update!([e1, e2, t1], "geometry", X)
    update!([e1, e2], "youngs modulus", 288.0)
    update!([e1, e2], "poissons ratio", 1/3)
    update!(t1, "displacement traction force", T)

    body1 = Problem(Elasticity, "block 1", 2)
    body1.properties.formulation = :plane_stress
    push!(body1, e1)
    body2 = Problem(Elasticity, "block 2", 2)
    body2.properties.formulation = :plane_stress
    push!(body2, e2, t1)

    # boundary elements for dirichlet dx=0
    dx1 = Element(Seg2, [1, 3])
    dx2 = Element(Seg2, [5, 7])
    update!([dx1, dx2], "geometry", X)
    update!([dx1, dx2], "displacement 1", 0.0)
    bc1 = Problem(Dirichlet, "symmetry dx=0", 2, "displacement")
    push!(bc1, dx1, dx2)

    # remove dof 9 from bc1 to avoid overconstrained problem
    push!(bc1.assembly.removed_dofs, 9)

    # boundary elements for dirichlet dy=0
    dy1 = Element(Seg2, [1, 2])
    update!(dy1, "geometry", X)
    update!(dy1, "displacement 2", 0.0)
    bc2 = Problem(Dirichlet, "symmetry dy=0", 2, "displacement")
    push!(bc2, dy1)

    # mortar boundary between two bodies
    mel1 = Element(Seg2, [3, 4])
    sel1 = Element(Seg2, [5, 6])
    update!([mel1, sel1], "geometry", X)
    bc3 = Problem(Mortar2D, "interface between blocks", 2, "displacement")
    add_slave_elements!(bc3, [sel1])
    add_master_elements!(bc3, [mel1])

    solver = LinearSolver(body1, body2, bc1, bc2, bc3)
    solver()

    u = e2("displacement", [1.0, 1.0], 0.0)
    u_expected = [-1/3, 1.0]
    @info("displacement at tip: $u")
    @test isapprox(u, u_expected)
end
