# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Postprocess
using JuliaFEM.Testing

@testset "test 2d linear elasticity with surface + volume load" begin
    meshfile = "/geometry/2d_block/BLOCK_1elem.med"
    mesh = aster_read_mesh(Pkg.dir("JuliaFEM")*meshfile)

    # field problem
    block = Problem(Elasticity, "BLOCK", 2)
    block.properties.formulation = :plane_strain
    block.properties.finite_strain = false
    block.properties.geometric_stiffness = false
    block.elements = create_elements(mesh, "BLOCK")
    update!(block.elements, "youngs modulus", 288.0)
    update!(block.elements, "poissons ratio", 1/3)

    # traction
    traction = Problem(Elasticity, "TRACTION", 2)
    traction.properties.formulation = :plane_strain
    traction.properties.finite_strain = false
    traction.properties.geometric_stiffness = false
    traction.elements = create_elements(mesh, "TOP")
    update!(traction, "displacement traction force 2", 288.0*9/8)

    # boundary conditions
    bc_sym_23 = Problem(Dirichlet, "symmetry bc 23", 2, "displacement")
    bc_sym_23.elements = create_elements(mesh, "LEFT")
    update!(bc_sym_23, "displacement 1", 0.0)
    bc_sym_13 = Problem(Dirichlet, "symmetry bc 13", 2, "displacement")
    bc_sym_13.elements = create_elements(mesh, "BOTTOM")
    update!(bc_sym_13, "displacement 2", 0.0)

    solver = LinearSolver(block, traction, bc_sym_23, bc_sym_13)
    solver()

    info("u = ", block.assembly.u)
    info("λ = ", block.assembly.la)

end

