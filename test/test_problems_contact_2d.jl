# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Postprocess
using JuliaFEM.Testing

testdir = joinpath(Pkg.dir("JuliaFEM"), "test")
datadir = first(splitext(basename(@__FILE__)))

@testset "hertz contact, full 2d model, linear elements, flat slave surface" begin
    meshfile = joinpath(testdir, datadir, "hertz_2d_full.med")
    mesh = aster_read_mesh(meshfile)

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

    nid = find_nearest_node(mesh, [0.0, 100.0])
    #load = Problem(Dirichlet, "load", 2, "displacement")
    load = Problem(Elasticity, "point load", 2)
    load.properties.formulation = :plane_strain
    load.elements = [Element(Poi1, [nid])]
    #update!(load.elements, "displacement 2", -10.0)
    update!(load, "displacement traction force 2", -35.0e3)

    contact = Problem(Contact2D, "contact between block and cylinder", 2, "displacement")
    contact.properties.rotate_normals = true
    contact_slave_elements = create_elements(mesh, "BLOCK_TO_CYLINDER")
    contact_master_elements = create_elements(mesh, "CYLINDER_TO_BLOCK")
    add_master_elements!(contact, contact_master_elements)
    add_slave_elements!(contact, contact_slave_elements)

    solver = Solver(Nonlinear)
    push!(solver, upper, lower, bc_fixed, bc_sym_23, load, contact)
    solver()

    node_ids, la = get_nodal_vector(contact_slave_elements, "lambda", 0.0)
    node_ids, n = get_nodal_vector(contact_slave_elements, "normal", 0.0)
    pres = [dot(ni, lai) for (ni, lai) in zip(n, la)]
    #@test isapprox(maximum(pres), 4060.010799583303)
    # 12 % error in maximum pressure
    # integrate pressure in normal and tangential direction
    Rn = 0.0
    Rt = 0.0
    Q = [0.0 -1.0; 1.0 0.0]
    time = 0.0
    for sel in contact_slave_elements
        for ip in get_integration_points(sel)
            w = ip.weight*sel(ip, time, Val{:detJ})
            n = sel("normal", ip, time)
            t = Q'*n
            la = sel("lambda", ip, time)
            Rn += w*dot(n, la)
            Rt += w*dot(t, la)
        end
    end
    info("2d hertz: Rn = $Rn, Rt = $Rt")
    info("2d hertz: maximum pressure pmax = ", maximum(pres))
    @test isapprox(maximum(pres), 3585.0; rtol = 0.13)
    # under 0.15 % error in resultant force
    @test isapprox(Rn, 35.0e3; rtol=0.020)
    @test isapprox(Rt, 0.0; atol=200.0)
end

function get_model()
    meshfile = joinpath(testdir, datadir, "block_2d.med")
    mesh = aster_read_mesh(meshfile)
    println(mesh.nodes[1])

    upper = Problem(mesh, Elasticity, "UPPER", 2)
    lower = Problem(mesh, Elasticity, "LOWER", 2)

    for body in [upper, lower]
        body.properties.formulation = :plane_stress
        update!(body, "youngs modulus", 288.0)
        update!(body, "poissons ratio", 1/3)
    end

    load = Problem(mesh, Elasticity, "UPPER_TOP", 2)
    load.properties.formulation = :plane_stress
    update!(load, "displacement traction force 2", -28.8)
    bc1 = Problem(mesh, Dirichlet, "LOWER_BOTTOM", 2, "displacement")
    update!(bc1, "displacement 2", 0.0)
    bc2 = Problem(mesh, Dirichlet, "LOWER_LEFT", 2, "displacement")
    update!(bc2, "displacement 1", 0.0)
    bc3 = Problem(mesh, Dirichlet, "UPPER_LEFT", 2, "displacement")
    update!(bc3, "displacement 1", 0.0)

    interface = Problem(Contact2D, "interface", 2, "displacement")
    interface.properties.rotate_normals = true
    interface_slave_elements = create_elements(mesh, "LOWER_TOP")
    interface_master_elements = create_elements(mesh, "UPPER_BOTTOM")
    add_master_elements!(interface, interface_master_elements)
    add_slave_elements!(interface, interface_slave_elements)

    # in LOWER_LEFT we have node belonging also to contact interface
    # let's remove it from dirichlet bc
    create_node_set_from_element_set!(mesh, "LOWER_LEFT")
    nid = find_nearest_node(mesh, [0.0, 0.5]; node_set="LOWER_LEFT")
    coords = mesh.nodes[nid]
    info("nearest node to (0.0, 0.5) = $nid, coordinates = $coords")
    dofs = [2*(nid-1)+1, 2*(nid-1)+2]
    info("removing nid $nid, dofs $dofs from LOWER_LEFT")
    push!(bc2.assembly.removed_dofs, dofs...)

    solver = Solver(Nonlinear)
    #push!(solver, upper, lower, load, bc1, interface)
    push!(solver, upper, lower, load, bc1, bc2, bc3, interface)
    return solver
end

@testset "small sliding 2d patch test, linear Seg2 elements, standard basis" begin

    solver = get_model()
    interface = solver["interface"]
    solver()

    node_ids, displacement = get_nodal_vector(interface.elements, "displacement", 0.0)
    node_ids, geometry = get_nodal_vector(interface.elements, "geometry", 0.0)
    node_ids, lambda = get_nodal_vector(interface.elements, "lambda", 0.0)
    u2 = [u[2] for u in displacement]
    f2 = [f[2] for f in lambda]
    maxabsu2 = maximum(abs.(u2))
    stdabsu2 = std(abs.(u2))
    info("max(abs(u2)) = $maxabsu2, std(abs(u2)) = $stdabsu2")
    @test isapprox(stdabsu2, 0.0; atol=1.0e-12)
    maxabsf2 = maximum(abs.(f2))
    stdabsf2 = std(abs.(f2))
    info("max(abs(f2)) = $maxabsf2, std(abs(f2)) = $stdabsf2")
    @test isapprox(stdabsf2, 0.0; atol=1.0e-12)
    @test isapprox(mean(abs.(f2)), 28.8; atol=1.0e-12)
end

@testset "small sliding 2d patch test, linear Seg2 elements, dual basis" begin

    solver = get_model()
    interface = solver["interface"]
    interface.properties.dual_basis = true
    solver()

    node_ids, displacement = get_nodal_vector(interface.elements, "displacement", 0.0)
    node_ids, geometry = get_nodal_vector(interface.elements, "geometry", 0.0)
    node_ids, lambda = get_nodal_vector(interface.elements, "lambda", 0.0)
    u2 = [u[2] for u in displacement]
    f2 = [f[2] for f in lambda]
    maxabsu2 = maximum(abs.(u2))
    stdabsu2 = std(abs.(u2))
    info("max(abs(u2)) = $maxabsu2, std(abs(u2)) = $stdabsu2")
    @test isapprox(stdabsu2, 0.0; atol=1.0e-12)
    maxabsf2 = maximum(abs.(f2))
    stdabsf2 = std(abs.(f2))
    info("max(abs(f2)) = $maxabsf2, std(abs(f2)) = $stdabsf2")
    @test isapprox(stdabsf2, 0.0; atol=1.0e-12)
    @test isapprox(mean(abs.(f2)), 28.8; atol=1.0e-12)
end
