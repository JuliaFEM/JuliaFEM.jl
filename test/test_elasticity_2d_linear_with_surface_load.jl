# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using Base.Test
using JuliaFEM

@testset "2d linear elasticity + volume load + surface load" begin

    X = Dict(1 => [0.0, 0.0],
             2 => [1.0, 0.0],
             3 => [1.0, 1.0],
             4 => [0.0, 1.0])

    props = ("formulation" => "plane_stress",
             "finite_strain" => "false",
             "geometric_stiffness" => "false")

    # field problem
    block = Problem(Elasticity, "BLOCK", 2)
    block.elements = [Element(Quad4, [1, 2, 3, 4])]
    update!(block.properties, props...)
    update!(block.elements, "geometry", X)
    update!(block.elements, "youngs modulus", 288.0)
    update!(block.elements, "poissons ratio", 1/3)
    update!(block.elements, "displacement load 2", 576.0)

    # traction
    traction = Problem(Elasticity, "TRACTION", 2)
    traction.elements = [Element(Seg2, [3, 4])]
    update!(traction.properties, props...)
    update!(traction.elements, "geometry", X)
    update!(traction.elements, "displacement traction force 2", 288.0)

    # boundary conditions
    bc = Problem(Dirichlet, "symmetry boundary conditions", 2, "displacement")
    bc.elements = [Element(Seg2, [1, 2]), Element(Seg2, [4, 1])]
    update!(bc.elements, "geometry", X)
    update!(bc.elements[1], "displacement 2", 0.0)
    update!(bc.elements[2], "displacement 1", 0.0)

    solver = Solver(Linear, "solve 2d linear elasticity problem")
    add_problems!(solver, [block, traction, bc])
    solve!(solver, 0.0)

    f = 288.0
    g = 576.0
    E = 288.0
    nu = 1/3
    u3 = block("displacement", 0.0)[3]
    u3_expected = f/E*[-nu, 1] + g/(2*E)*[-nu, 1]
    @test isapprox(u3, u3_expected)
end
