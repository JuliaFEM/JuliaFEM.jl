# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Postprocess
using JuliaFEM.Test
import JuliaFEM: get_mesh, get_model

function get_mesh(::Type{Val{Symbol("curved 2d mesh model")}})
    meshfile = Pkg.dir("JuliaFEM") * "/test/testdata/block_2d_curved.med"
    mesh = aster_read_mesh(meshfile)
end

function get_model(::Type{Val{Symbol("curved 2d contact small sliding")}})
    
    mesh = get_mesh("curved 2d mesh model")

    upper = Problem(Elasticity, "upper", 2)
    upper.properties.formulation = :plane_stress
    upper.elements = create_elements(mesh, "UPPER")
    update!(upper, "youngs modulus", 96.0)
    update!(upper, "poissons ratio", 1/3)

    lower = Problem(Elasticity, "lower", 2)
    lower.properties.formulation = :plane_stress
    lower.elements = create_elements(mesh, "LOWER")
    update!(lower, "youngs modulus", 96.0)
    update!(lower, "poissons ratio", 1/3)

    bc_upper = Problem(Dirichlet, "upper boundary", 2, "displacement")
    bc_upper.elements = create_elements(mesh, "UPPER_TOP")
    update!(bc_upper, "displacement 1", 0.0)
    update!(bc_upper, "displacement 2", -0.15)

    bc_lower = Problem(Dirichlet, "lower boundary", 2, "displacement")
    bc_lower.elements = create_elements(mesh, "LOWER_BOTTOM")
    update!(bc_lower, "displacement 1", 0.0)
    update!(bc_lower, "displacement 2", 0.0)

    interface = Problem(Contact, "contact between upper and lower block", 2, "displacement")
    interface.properties.dimension = 1
    interface.properties.rotate_normals = true
    interface_slave_elements = create_elements(mesh, "LOWER_TOP")
    interface_master_elements = create_elements(mesh, "UPPER_BOTTOM")
    update!(interface_slave_elements, "master elements", interface_master_elements)
    interface.elements = [interface_master_elements; interface_slave_elements]
    info("type of list is ", typeof(first(interface_slave_elements)("master elements", 0.0)))

    solver = Solver(Nonlinear)
    push!(solver, upper, lower, bc_upper, bc_lower, interface)
    return solver

end

@testset "test all nodes in contact" begin
    # FIXME: needs verification of some other fem software
    solver = get_model("curved 2d contact small sliding")
    call(solver)
    upper, lower, bc_upper, bc_lower, interface = solver.problems
    @test isapprox(norm(interface.assembly.u), 0.49563347601324315)
end


function get_mesh(::Type{Val{Symbol("hertz contact, full 2d model")}})
    meshfile = Pkg.dir("JuliaFEM") * "/test/testdata/hertz_2d_full.med"
    mesh = aster_read_mesh(meshfile)
end

function get_model(::Type{Val{Symbol("hertz contact, full 2d model")}})
    # from fenet d3613 advanced finite element contact benchmarks
    # a = 6.21 mm, pmax = 3585 MPa
    # this is a very dense mesh and for that reason pmax is not very
    # (only 6 elements in -20 .. 20 mm contact zone, 3 elements in contact
    # instead integrate pressure in normal and tangential direction
    mesh = get_mesh("hertz contact, full 2d model")

    upper = Problem(Elasticity, "CYLINDER", 2)
    upper.properties.formulation = :plane_strain
    upper.elements = create_elements(mesh, "CYLINDER")
    update!(upper, "youngs modulus", 70.0e3)
    update!(upper, "poissons ratio", 0.3)

    lower = Problem(Elasticity, "BLOCK", 2)
    lower.properties.formulation = :plane_strain
    lower.elements = create_elements(mesh, "BLOCK")
    update!(lower, "youngs modulus", 210.0e3)
    update!(lower, "poissons ratio", 0.3)

    # support block to ground
    bc_fixed = Problem(Dirichlet, "fixed", 2, "displacement")
    bc_fixed.elements = create_elements(mesh, "FIXED")
    update!(bc_fixed, "displacement 2", 0.0)

    # symmetry line
    bc_sym_23 = Problem(Dirichlet, "symmetry line 23", 2, "displacement")
    bc_sym_23.elements = create_elements(mesh, "SYM23")
    update!(bc_sym_23, "displacement 1", 0.0)

    nid = find_nearest_nodes(mesh, [0.0, 100.0])
    #load = Problem(Dirichlet, "load", 2, "displacement")
    load = Problem(Elasticity, "point load", 2)
    load.properties.formulation = :plane_strain
    load.elements = [Element(Poi1, nid)]
    #update!(load.elements, "displacement 2", -10.0)
    update!(load, "displacement traction force 2", -35.0e3)

    contact = Problem(Contact, "contact between block and cylinder", 2, "displacement")
    contact.properties.rotate_normals = true
    contact_slave_elements = create_elements(mesh, "CYLINDER_TO_BLOCK")
    contact_master_elements = create_elements(mesh, "BLOCK_TO_CYLINDER")
    update!(contact_slave_elements, "master elements", contact_master_elements)
    contact.elements = [contact_master_elements; contact_slave_elements]
    
    solver = Solver(Nonlinear)
    push!(solver, upper, lower, bc_fixed, bc_sym_23, load, contact)
    return solver

end

@testset "test frictionless hertz contact, 2d plane strain" begin
    solver = get_model("hertz contact, full 2d model")
    call(solver)
    upper, lower, bc_fixed, bc_sym_23, load, contact = solver.problems
    slaves = get_slave_elements(contact)
    node_ids, la = get_nodal_vector(slaves, "reaction force", 0.0)
    node_ids, n = get_nodal_vector(slaves, "normal", 0.0)
    pres = [dot(ni, lai) for (ni, lai) in zip(n, la)]
    @test isapprox(maximum(pres), 4060.010799583303)
    # integrate pressure in normal and tangential direction
    Rn = 0.0
    Rt = 0.0
    Q = [0.0 -1.0; 1.0 0.0]
    time = 0.0
    for sel in slaves
        for ip in get_integration_points(sel)
            w = ip.weight*sel(ip, time, Val{:detJ})
            n = sel("normal", ip, time)
            t = Q'*n
            la = sel("reaction force", ip, time)
            Rn += w*dot(n, la)
            Rt += w*dot(t, la)
        end
    end
    @test isapprox(Rn, 35.0e3; rtol=0.0015)
    @test isapprox(Rt, 0.0; atol=10.0)
end

