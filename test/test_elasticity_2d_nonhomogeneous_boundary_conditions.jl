# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Test

@testset "2d nonlinear elasticity: test nonhomogeneous boundary conditions and stress calculation" begin

    # field problem
    block = Problem(Elasticity, "BLOCK", 2)
    block.properties.formulation = :plane_stress
    block.properties.finite_strain = true
    block.properties.geometric_stiffness = true

    nodes = Dict{Int, Vector{Float64}}(
        1 => [0.0, 0.0],
        2 => [1.0, 0.0],
        3 => [1.0, 1.0],
        4 => [0.0, 1.0])

    element = Element(Quad4, [1, 2, 3, 4])
    update!(element, "geometry", nodes)
    update!(element, "youngs modulus", 288.0)
    update!(element, "poissons ratio", 1/3)
    push!(block, element)

    # boundary conditions
    bc = Problem(Dirichlet, "bc", 2, "displacement")
    bel1 = Element(Seg2, [1, 2])
    bel2 = Element(Seg2, [3, 4])
    bel3 = Element(Seg2, [4, 1])
    update!([bel1, bel2, bel3], "geometry", nodes)
    update!(bel1, "displacement 2", 0.0)
    update!(bel2, "displacement 2", 0.5)
    update!(bel3, "displacement 1", 0.0)
    push!(bc, bel1, bel2, bel3)

    solver = Solver("solve block problem")
    push!(solver, block, bc)
    call(solver)

    # from code aster
    eps_expected = [-2.08333312468287E-01, 6.25000000000000E-01, 0.0]
    sig_expected = [ 4.50685020821470E-06, 4.62857140373777E+02, 0.0]
    u3_expected =  [-2.36237356855269E-01, 5.00000000000000E-01]

    u3 = reshape(block.assembly.u, 2, 4)[:, 3]
    info("u3 = $u3")
    @test isapprox(u3, u3_expected, atol=1.0e-5)

    info("strain")
    for ip in get_integration_points(element)
        eps = ip("strain")
        @printf "%i | %8.3f %8.3f | %8.3f %8.3f %8.3f\n" ip.id ip.coords[1] ip.coords[2] eps[1] eps[2] eps[3]
        @test isapprox(eps, eps_expected, atol=1.0e-5)
    end
#=
    info("cauchy stress")
    for ip in get_integration_points(element)
        sig = ip("cauchy stress")
        @printf "%i | %8.3f %8.3f | %8.3f %8.3f %8.3f\n" ip.id ip.coords[1] ip.coords[2] sig[1] sig[2] sig[3]
        @test isapprox(sig, sig_expected)
    end
    info("pk2 stress")
    for ip in get_integration_points(element)
        sig = ip("pk2 stress")
        @printf "%i | %8.3f %8.3f | %8.3f %8.3f %8.3f\n" ip.id ip.coords[1] ip.coords[2] sig[1] sig[2] sig[3]
        @test isapprox(sig, sig_expected)
    end
=#
end

