# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Postprocess
using JuliaFEM.Testing

#=
- solve 2d plane stress problem with known solution:
    surface traction force in 2d
    volume load in 2d
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
    traction = Problem(Elasticity, "TRACTION", 2)
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

    solver = Solver(Linear, block, traction, bc_sym_23, bc_sym_13)
    solver()

    info("u = ", block.assembly.u)
    info("λ = ", block.assembly.la)

    f = 288.0
    g = 576.0
    E = 288.0
    nu = 1/3
    u3 = block("displacement", 0.0)[3]
    u3_expected = f/E*[-nu, 1] + g/(2*E)*[-nu, 1]
    @test isapprox(u3, u3_expected)
end

print_statistics()
