# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Testing

@testset "test 2d nonlinear elasticity with surface load" begin

    X = Dict(1 => [0.0, 0.0],
             2 => [1.0, 0.0],
             3 => [1.0, 1.0],
             4 => [0.0, 1.0])

    el1 = Element(Quad4, [1, 2, 3, 4])
    update!(el1, "geometry", X)
    update!(el1, "youngs modulus", 288.0)
    update!(el1, "poissons ratio", 1/3)
    update!(el1, "displacement load 2", 576.0)

    el2 = Element(Seg2, [3, 4])
    update!(el2, "geometry", X)
    update!(el2, "displacement traction force 2", 288.0)

    # field problem
    block = Problem(Elasticity, "BLOCK", 2)
    block.properties.formulation = :plane_stress
    block.properties.finite_strain = true
    block.properties.geometric_stiffness = true
    push!(block.elements, el1, el2)

    el3 = Element(Seg2, [1, 2])
    update!(el3, "geometry", X)
    update!(el3, "displacement 2", 0.0)
    el4 = Element(Seg2, [4, 1])
    update!(el4, "geometry", X)
    update!(el4, "displacement 1", 0.0)

    # boundary conditions
    bc_sym = Problem(Dirichlet, "symmetry bc", 2, "displacement")
    push!(bc_sym, el3, el4)

    solver = Solver(Nonlinear, block, bc_sym)
    solver()

    # verified using from code aster
    u3_expected = [-4.92316106779943E-01, 7.96321884292103E-01]
    eps_zz = -3.71128811855451E-01
    eps_expected = [-3.71128532282463E-01, 1.11338615599337E+00, 0.0]
    sig_expected = [ 3.36174888827909E-05, 2.23478729403118E+03, 0.0]

    u3 = reshape(get_solution_vector(solver), 2, 4)[:,3]
    info("u3 = $u3, u3_expected = $u3_expected")
    @test isapprox(u3, u3_expected, atol=1.0e-5)
end

print_statistics()
