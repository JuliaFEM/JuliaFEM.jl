# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Test

@testset "test 2d linear elasticity with surface load" begin
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
#   update!(block.elements, "displacement load 2", 576.0)

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

    solver = Solver("solve block problem")
    push!(solver, block, bc_sym)
    call(solver)

    f = 288.0
    g = 576.0
    E = 288.0
    nu = 1/3
    u3_expected = f/E*[-nu, 1] + g/(2*E)*[-nu, 1]
    u3 = reshape(block.assembly.u, 2, 4)[:,3]
    info("u3 = $u3")
    @test isapprox(u3, u3_expected)

    info("strain")
    for ip in get_integration_points(block.elements[1])
        eps = ip("strain")
        @printf "%i | %8.3f %8.3f | %8.3f %8.3f %8.3f\n" ip.id ip.coords[1] ip.coords[2] eps[1] eps[2] eps[3]
        @test isapprox(eps, [u3; 0.0])
    end

    info("stress")
    for ip in get_integration_points(block.elements[1])
        sig = ip("stress")
        @printf "%i | %8.3f %8.3f | %8.3f %8.3f %8.3f\n" ip.id ip.coords[1] ip.coords[2] sig[1] sig[2] sig[3]
        @test isapprox(sig, [0.0; g; 0.0])
    end

end
