# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM, Test

#=
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

    plastic_parameters = Dict{Any, Any}("type" => JuliaFEM.ideal_plasticity!,
                                        "yield_surface" => Val{:von_mises},
                                        "params" => Dict("yield_stress" => 175.0))
    to_integ_points = Dict()
    map(x-> to_integ_points[x] = plastic_parameters, get_connectivity(element))
    update!(element, "plasticity", to_integ_points)
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

    solver = NonlinearSolver("solve block problem")
    solver.time = 1.0
    push!(solver, block, bc)
    solver()

    # from code aster
    eps_expected = [-2.08333312468287E-01, 6.25000000000000E-01, 0.0]
    sig_expected = [ 4.50685020821470E-06, 4.62857140373777E+02, 0.0]
    u3_expected =  [-2.36237356855269E-01, 5.00000000000000E-01]

    u3 = reshape(block.assembly.u, 2, 4)[:, 3]
    @info("u3 = $u3")
    @test isapprox(u3, u3_expected, atol=1.0e-5)
end

=#
