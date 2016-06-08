# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Test

@testset "test 2d nonlinear elasticity with surface load" begin
    meshfile = "/geometry/2d_block/BLOCK_1elem.med"
    mesh = parse_aster_med_file(Pkg.dir("JuliaFEM")*meshfile)

    # field problem
    block = Problem(Elasticity, "BLOCK", 2)
    block.properties.formulation = :plane_stress
    block.properties.finite_strain = true

    elements = aster_create_elements(mesh, :BLOCK, :QU4)
    update!(elements, "youngs modulus", 288.0)
    update!(elements, "poissons ratio", 1/3)
    update!(elements, "displacement load 2", 576.0)
    push!(block, elements...)

    traction = aster_create_elements(mesh, :TOP, :SE2)
    update!(traction, "displacement traction force 2", 288.0)
    push!(block, traction...)

    # boundary conditions
    bc_sym = Problem(Dirichlet, "symmetry bc", 2, "displacement")
    bc_elements_left = aster_create_elements(mesh, :LEFT, :SE2)
    bc_elements_bottom = aster_create_elements(mesh, :BOTTOM, :SE2)
    update!(bc_elements_left, "displacement 1", 0.0)
    update!(bc_elements_bottom, "displacement 2", 0.0)
    push!(bc_sym, bc_elements_left..., bc_elements_bottom...)

    solver = Solver("solve block problem")
    push!(solver, block, bc_sym)
    call(solver)

    # from code aster
    u3_expected = [-4.92316106779943E-01, 7.96321884292103E-01]
    eps_zz = -3.71128811855451E-01
    eps_expected = [-3.71128532282463E-01, 1.11338615599337E+00, 0.0]
    sig_expected = [ 3.36174888827909E-05, 2.23478729403118E+03, 0.0]

    u3 = reshape(block.assembly.u, 2, 4)[:, 3]
    info("u3 = $u3")
    @test isapprox(u3, u3_expected, atol=1.0e-5)

    info("strain")
    for ip in get_integration_points(elements[1])
        eps = ip("strain")
        #eps = [eps[1,1]; eps[2,2]; eps[1,2]]
        @printf "%i | %8.3f %8.3f | %8.3f %8.3f %8.3f\n" ip.id ip.coords[1] ip.coords[2] eps[1] eps[2] eps[3]
        @test isapprox(eps, eps_expected)
    end

    info("stress")
    for ip in get_integration_points(elements[1])
        sig = ip("stress")
        #sig = [sig[1,1]; sig[2,2]; sig[1,2]]
        @printf "%i | %8.3f %8.3f | %8.3f %8.3f %8.3f\n" ip.id ip.coords[1] ip.coords[2] sig[1] sig[2] sig[3]
        @test isapprox(sig, sig_expected)
    end
end
