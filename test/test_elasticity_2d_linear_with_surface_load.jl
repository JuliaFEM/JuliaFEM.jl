# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Postprocess
using JuliaFEM.Testing

#=
- solve 2d plane stress problem with known solution
- test postprocessing of nodal fields: (geometry, displacement
  reaction force, concentrated force)
=#
@testset "test 2d linear elasticity with surface + volume load" begin
    meshfile = "/geometry/2d_block/BLOCK_1elem.med"
    mesh = aster_read_mesh(Pkg.dir("JuliaFEM")*meshfile)

    # field problem
    block = Problem(Elasticity, "BLOCK", 2)
    block.properties.formulation = :plane_stress
    block.properties.finite_strain = false
    block.properties.geometric_stiffness = false
    block.elements = create_elements(mesh, "BLOCK")
    update!(block.elements, "youngs modulus", 288.0)
    update!(block.elements, "poissons ratio", 1/3)
    update!(block.elements, "displacement load 2", 576.0)

    # traction
    traction = Problem(Elasticity, "BLOCK", 2)
    traction.properties.formulation = :plane_stress
    traction.properties.finite_strain = false
    traction.properties.geometric_stiffness = false
    traction.elements = create_elements(mesh, "TOP")
    update!(traction, "displacement traction force 2", 288.0)

    # boundary conditions
    bc_sym_23 = Problem(Dirichlet, "symmetry bc 23", 2, "displacement")
    bc_sym_23.elements = create_elements(mesh, "LEFT")
    update!(bc_sym_23, "displacement 1", 0.0)
    bc_sym_13 = Problem(Dirichlet, "symmetry bc 13", 2, "displacement")
    bc_sym_13.elements = create_elements(mesh, "BOTTOM")
    update!(bc_sym_13, "displacement 2", 0.0)

    solver = LinearSolver(block, traction, bc_sym_23, bc_sym_13)
    solver()

    f = 288.0
    g = 576.0
    E = 288.0
    nu = 1/3
    u3_expected = f/E*[-nu, 1] + g/(2*E)*[-nu, 1]

    # fetch nodal results X + u and join them into one table using DataFrames
    X = block(DataFrame, "geometry", :COOR, 0.0)
    u = block(DataFrame, "displacement", :U, 0.0)
    results = join(X, u, on=:id, kind=:outer)
    println(results)

    u3 = results[:N3, [:U1, :U2]]
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
