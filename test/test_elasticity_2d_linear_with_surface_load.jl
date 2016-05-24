# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Test

@testset "test 2d linear elasticity with surface load" begin
    meshfile = "/geometry/2d_block/BLOCK_1elem.med"
    mesh = parse_aster_med_file(Pkg.dir("JuliaFEM")*meshfile)

    # field problem
    block = Problem(Elasticity, "BLOCK", 2)
    block.properties.formulation = :plane_stress
    block.properties.finite_strain = false

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
    f = 288.0
    g = 576.0
    E = 288.0
    nu = 1/3
    u3_expected = f/E*[-nu, 1] + g/(2*E)*[-nu, 1]
    u3 = reshape(block.assembly.u, 2, 4)[:,3]
    @test isapprox(u3, u3_expected)
end
