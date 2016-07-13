# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Postprocess
using JuliaFEM.Testing

function JuliaFEM.get_model(::Type{Val{Symbol("test 2d linear elasticity with surface + volume load")}})
    meshfile = "/geometry/2d_block/BLOCK_1elem.med"
    mesh = aster_read_mesh(Pkg.dir("JuliaFEM")*meshfile)

    # field problem
    block = Problem(Elasticity, "BLOCK", 2)
    block.properties.store_fields = ["stress", "strain"]
    block.properties.formulation = :plane_stress
    block.properties.finite_strain = false
    block.properties.geometric_stiffness = false

    block.elements = create_elements(mesh, "BLOCK")
    update!(block.elements, "youngs modulus", 288.0)
    update!(block.elements, "poissons ratio", 1/3)
    update!(block.elements, "displacement load 2", 576.0)

    traction = create_elements(mesh, "TOP")
    update!(traction, "displacement traction force 2", 288.0)
    push!(block, traction...)

    # boundary conditions
    bc_sym = Problem(Dirichlet, "symmetry bc", 2, "displacement")
    bc_elements_left = create_elements(mesh, "LEFT")
    bc_elements_bottom = create_elements(mesh, "BOTTOM")
    update!(bc_elements_left, "displacement 1", 0.0)
    update!(bc_elements_bottom, "displacement 2", 0.0)
    push!(bc_sym, bc_elements_left..., bc_elements_bottom...)

    solver = LinearSolver("solve block problem")
    push!(solver, block, bc_sym)
    return solver
end

@testset "test 2d linear elasticity with surface + volume load" begin

    solver = get_model("test 2d linear elasticity with surface + volume load")
    block, bc_sym = solver.problems
    solver()

    f = 288.0
    g = 576.0
    E = 288.0
    nu = 1/3
    u3_expected = f/E*[-nu, 1] + g/(2*E)*[-nu, 1]

    results = block(DataFrame, "displacement", :U, 0.0)
    println(results)
    u3 = results[:N3, [:U1, :U2]]
    info("(u1,u2) at node 3")
    info(u3)
    @test isapprox(u3, u3_expected)

#=
    info("strain")
    for ip in get_integration_points(block.elements[1])
        eps = ip("strain")
        @printf "%i | %8.3f %8.3f | %8.3f %8.3f %8.3f\n" ip.id ip.coords[1] ip.coords[2] eps[1] eps[2] eps[3]
        # TODO: to postprocess ...?
        #@test isapprox(eps, [u3[1], u3[2], 0.0])
    end

    info("stress")
    for ip in get_integration_points(block.elements[1])
        sig = ip("stress")
        @printf "%i | %8.3f %8.3f | %8.3f %8.3f %8.3f\n" ip.id ip.coords[1] ip.coords[2] sig[1] sig[2] sig[3]
        # TODO: to postprocess
        #@test isapprox(sig, [0.0, g, 0.0])
    end

    calc_nodal_values!(block.elements, "strain", 3, 0.0)
    calc_nodal_values!(block.elements, "stress", 3, 0.0)
    info(block.elements[1]["stress"](0.0))
    node_ids, strain = get_nodal_vector(block.elements, "strain", 0.0)
    node_ids, stress = get_nodal_vector(block.elements, "stress", 0.0)
    # TODO: to postprocess
    #@test isapprox(stress[1], [0.0, g, 0.0])
    #@test isapprox(strain[1], [u3[1], u3[2], 0.0])
=#

end

#= TODO: to other file
@testset "test dump model to disk and read back before and after solution" begin
    solver = get_model("test 2d linear elasticity with surface + volume load")
    save("/tmp/model.jld", "linear_model", solver)
    solver2 = load("/tmp/model.jld")["linear_model"]
    solver2()
    save("/tmp/model.jld", "results", solver2)
    solver3 = load("/tmp/model.jld")["results"]
    block = solver3["BLOCK"]
    u3 = reshape(block.assembly.u, 2, 4)[:,3]
    f = 288.0
    g = 576.0
    E = 288.0
    nu = 1/3
    u3_expected = f/E*[-nu, 1] + g/(2*E)*[-nu, 1]
    @test isapprox(u3, u3_expected)
end
=#
