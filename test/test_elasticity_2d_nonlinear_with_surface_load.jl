# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Test

@testset "test 2d nonlinear elasticity with surface load" begin
    meshfile = "/geometry/2d_block/BLOCK_1elem.med"
    mesh = parse_aster_med_file(Pkg.dir("JuliaFEM")*meshfile)
    # field problem
    body = Problem(Elasticity, "BLOCK", 2)
    body.properties.formulation = :plane_stress
    body_elements = aster_create_elements(mesh, :BLOCK, :QU4)
    update!(body_elements, "youngs modulus", 900.0)
    update!(body_elements, "poissons ratio", 0.25)
    trac_elements = aster_create_elements(mesh, :TOP, :SE2)
    update!(trac_elements, "displacement traction force 2", -100.0)
    push!(body, body_elements..., trac_elements...)
    # boundary conditions
    bc_sym = Problem(Dirichlet, "symmetry bc", 2, "displacement")
    bc_elements_left = aster_create_elements(mesh, :LEFT, :SE2)
    bc_elements_bottom = aster_create_elements(mesh, :BOTTOM, :SE2)
    update!(bc_elements_left, "displacement 1", 0.0)
    update!(bc_elements_bottom, "displacement 2", 0.0)
    push!(bc_sym, bc_elements_left..., bc_elements_bottom...)
    solver = Solver("solve block problem")
    push!(solver, body, bc_sym)
    call(solver)
    # result is verified using code aster
    u3_expected = [3.17431158889468E-02, -1.38591518927826E-01]
    u3 = reshape(body.assembly.u, 2, 4)[:,3]
    @test isapprox(u3, u3_expected)
end
