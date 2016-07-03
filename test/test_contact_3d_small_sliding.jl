# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Postprocess
using JuliaFEM.Test

@testset "3d upper side curved contact" begin

    # TODO: accurate solution is not known, verify using another fem software
    # however results look very meaningful and probably this is right.

    meshfile = Pkg.dir("JuliaFEM") * "/test/testdata/block_3d_curved.med"
    mesh = aster_read_mesh(meshfile)

    upper = Problem(Elasticity, "upper", 3)
    upper.elements = create_elements(mesh, "UPPER")
    update!(upper, "youngs modulus", 96.0)
    update!(upper, "poissons ratio", 1/3)

    lower = Problem(Elasticity, "lower", 3)
    lower.elements = create_elements(mesh, "LOWER")
    update!(lower, "youngs modulus", 96.0)
    update!(lower, "poissons ratio", 1/3)

    bc_upper = Problem(Dirichlet, "upper boundary", 3, "displacement")
    bc_upper.elements = create_elements(mesh, "UPPER_TOP")
    update!(bc_upper, "displacement 1",  0.0)
    update!(bc_upper, "displacement 2",  0.0)
    update!(bc_upper, "displacement 3", -0.1)

    bc_lower = Problem(Dirichlet, "lower boundary", 3, "displacement")
    bc_lower.elements = create_elements(mesh, "LOWER_BOTTOM")
    update!(bc_lower, "displacement 1", 0.0)
    update!(bc_lower, "displacement 2", 0.0)
    update!(bc_lower, "displacement 3", 0.0)

    contact = Problem(Contact, "contact between upper and lower block", 3, "displacement")
    contact_slave_elements = create_elements(mesh, "LOWER_TOP")
    contact_master_elements = create_elements(mesh, "UPPER_BOTTOM")
    update!(contact_slave_elements, "master elements", contact_master_elements)
    contact.elements = [contact_master_elements; contact_slave_elements]

    solver = Solver(Nonlinear)
    push!(solver, upper, lower, bc_upper, bc_lower, contact)

    call(solver)
    for element in get_slave_elements(contact)
        normal = element("normal", [1/3, 1/3], solver.time)
        @test isapprox(normal, [0.0, 0.0, 1.0])
        pres = dot(normal, element("reaction force", [1/3, 1/3], solver.time))
        info("pressure = $pres")
        #info(element("displacement", [1/3, 1/3], solver.time))
    end
    normu = norm(contact.assembly.u)
    info("displacement field norm = $normu")
    @test isapprox(normu, 0.7417557629004985)

end
