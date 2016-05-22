# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Test

@testset "test simple continuum block with surface traction" begin

    nodes = Dict{Int64, Node}(
        1 => [0.0, 0.0, 0.0],
        2 => [1.0, 0.0, 0.0],
        3 => [1.0, 1.0, 0.0],
        4 => [0.0, 1.0, 0.0],
        5 => [0.0, 0.0, 1.0],
        6 => [1.0, 0.0, 1.0],
        7 => [1.0, 1.0, 1.0],
        8 => [0.0, 1.0, 1.0])
    element1 = Element(Hex8, [1, 2, 3, 4, 5, 6, 7, 8])
    element2 = Element(Quad4, [5, 6, 7, 8])
    update!([element1, element2], "geometry", nodes)
    update!(element1, "youngs modulus", 900.0)
    update!(element1, "poissons ratio", 0.25)
    element2["displacement traction force"] = Vector{Float64}[[0.0, 0.0, -100.0] for i=1:4]

    problem = Problem(Elasticity, "block", 3)
    push!(problem, element1, element2)

#=
    free_dofs = zeros(Bool, 8, 3)
    x = 1
    y = 2
    z = 3
    free_dofs[2, x] = true
    free_dofs[3, [x, y]] = true
    free_dofs[4, y] = true
    free_dofs[5, z] = true
    free_dofs[6, [x, z]] = true
    free_dofs[7, [x, y, z]] = true
    free_dofs[8, [y, z]] = true
    free_dofs = find(vec(free_dofs'))
    info("free dofs: $free_dofs")

    ass = assemble(problem, 0.0)
    f = full(ass.force_vector)
    K = full(ass.stiffness_matrix)
#   info("initial force vector")
#   dump(reshape(f, 3, 8))
#   info("initial stiffness matrix")
#   dump(round(Int, K)[free_dofs, free_dofs])
    u = zeros(3, 8)
    u[free_dofs] = K[free_dofs, free_dofs] \ f[free_dofs]
    info("result vector")
    dump(u)
=#

    dx = Element(Quad4, [1, 4, 8, 5])
    dx["displacement 1"] = 0.0
    dy = Element(Quad4, [1, 5, 6, 2])
    dy["displacement 2"] = 0.0
    dz = Element(Quad4, [1, 2, 3, 4])
    dz["displacement 3"] = 0.0
    bc = Problem(Dirichlet, "symmetries", 3, "displacement")
    update!([dx, dy, dz], "geometry", nodes)
    push!(bc, dx, dy, dz)

    solver = Solver()
    push!(solver, problem, bc)
    solver()

    X = element1("geometry", [1.0, 1.0, 1.0], 0.0)
    u = element1("displacement", [1.0, 1.0, 1.0], 0.0)
    info("displacement at $X = $u")

    # verified using Code Aster.
    # 2015-12-12-continuum-elasticity/c3d_linear.*
    # [1/36, 1/36, -1/9]
    @test isapprox(u, [2.77777777777778E-02, 2.77777777777778E-02, -1.11111111111111E-01])
end

