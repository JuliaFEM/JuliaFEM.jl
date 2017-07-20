# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Postprocess
using JuliaFEM.Testing

tet4_meshfile = "test_problems_contact_3d/tet4.inp"
tet10_meshfile = "test_problems_contact_3d/tet10.inp"

function get_model(meshfile)
    mesh = abaqus_read_mesh(meshfile)
    
    upper = Problem(Elasticity, "UPPER", 3)
    upper.elements = create_elements(mesh, "UPPER")
    update!(upper, "youngs modulus", 3*288.0)
    update!(upper, "poissons ratio", 1/3)

    lower = Problem(Elasticity, "LOWER", 3)
    lower.elements = create_elements(mesh, "LOWER")
    update!(lower, "youngs modulus", 288.0)
    update!(lower, "poissons ratio", 1/3)

    bc_upper = Problem(Dirichlet, "UPPER_TOP", 3, "displacement")
    bc_upper.elements = create_surface_elements(mesh, "UPPER_TOP")
    update!(bc_upper, "displacement 3", -0.4)

    bc_lower = Problem(Dirichlet, "LOWER_BOTTOM", 3, "displacement")
    bc_lower.elements = create_surface_elements(mesh, "LOWER_BOTTOM")
    update!(bc_lower, "displacement 3", 0.0)
    
    # point-wise boundary conditions to prevent free body move
    nid1 = find_nearest_nodes(mesh, [0.0, 0.0, 0.0])[1]
    nid2 = find_nearest_nodes(mesh, [1.0, 0.0, 0.0])[1]
    nid3 = find_nearest_nodes(mesh, [0.0, 1.0, 0.0])[1]
    nid4 = find_nearest_nodes(mesh, [0.0, 0.0, 1.0])[1]
    nid5 = find_nearest_nodes(mesh, [1.0, 0.0, 1.0])[1]
    nid6 = find_nearest_nodes(mesh, [0.0, 1.0, 1.0])[1]

    bc_sym13 = Problem(Dirichlet, "SYM13", 3, "displacement")
    # nodes in X2=0 plane
    bc_sym13.elements = [Element(Poi1, [j]) for j in [nid1, nid2, nid4, nid5]]
    update!(bc_sym13, "geometry", mesh.nodes)
    update!(bc_sym13, "displacement 2", 0.0)

    bc_sym23 = Problem(Dirichlet, "SYM23", 3, "displacement")
    # nodes in X1=0 plane
    bc_sym23.elements = [Element(Poi1, [j]) for j in [nid1, nid3, nid4, nid6]]
    update!(bc_sym23, "geometry", mesh.nodes)
    update!(bc_sym23, "displacement 1", 0.0)

    interface = Problem(Contact, "LOWER_TO_UPPER", 3, "displacement")
    interface_slave_elements = create_surface_elements(mesh, "LOWER_TO_UPPER")
    interface_master_elements = create_surface_elements(mesh, "UPPER_TO_LOWER")
    update!(interface_slave_elements, "master elements", interface_master_elements)
    interface.elements = [interface_slave_elements; interface_master_elements]
    interface.properties.contact_state_in_first_iteration = :AUTO
    
    #append!(bc_sym13.assembly.removed_dofs, [1316, 1319, 1388, 1358])
    #append!(bc_sym23.assembly.removed_dofs, [1492, 1627, 1387, 1597])
    #append!(interface.assembly.removed_dofs, [1316, 1319, 1358, 1387, 1388, 1492, 1597, 1627])

    solver = NonlinearSolver(upper, lower, bc_upper, bc_lower, bc_sym13, bc_sym23, interface)
#    solver.properties.max_iterations = 5

    return solver
end

@testset "small sliding contact patch test, tet4 + standard basis" begin
    solver = get_model(tet4_meshfile)
    solver.xdmf = Xdmf("contact_sl_lin_disp_results"; overwrite=true)
    interface = solver["LOWER_TO_UPPER"]
    interface.properties.dual_basis = false
    solver()
    node_ids, displacement = get_nodal_vector(interface.elements, "displacement", 0.0)
    node_ids, geometry = get_nodal_vector(interface.elements, "geometry", 0.0)
    # test postprocess of fields
    postprocess!(interface, 0.0, Val{Symbol("contact pressure")})
    node_ids, contact_pressure = get_nodal_vector(interface.elements, "contact pressure", 0.0)
    u3 = [u[3] for u in displacement]
    maxabsu3 = maximum(abs(u3))
    stdabsu3 = std(abs(u3))
    maxpres = maximum(contact_pressure)
    stdpres = std(contact_pressure)
    info("max(abs(u3)) = $maxabsu3, std(abs(u3)) = $stdabsu3")
    info("max(contact_pressure) = $maxpres, std(contact_pressure) = $stdpres")
    @test isapprox(stdabsu3, 0.0; atol=1.0e-12)
    @test isapprox(maxpres, 172.8; atol=1.0e-6)
end

@testset "small sliding contact patch test, tet4 + dual basis" begin
    solver = get_model(tet4_meshfile)
    solver.xdmf = Xdmf("contact_dl_lin_disp_results"; overwrite=true)
    interface = solver["LOWER_TO_UPPER"]
    interface.properties.dual_basis = true
    solver()
    node_ids, displacement = get_nodal_vector(interface.elements, "displacement", 0.0)
    node_ids, geometry = get_nodal_vector(interface.elements, "geometry", 0.0)
    u3 = [u[3] for u in displacement]
    maxabsu3 = maximum(abs(u3))
    stdabsu3 = std(abs(u3))
    info("max(abs(u3)) = $maxabsu3, std(abs(u3)) = $stdabsu3")
    @test isapprox(stdabsu3, 0.0; atol=1.0e-12)
end

@testset "small sliding contact patch test, tet10 + standard basis" begin
    solver = get_model(tet10_meshfile)
    solver.xdmf = Xdmf("contact_sl_quad_disp_results"; overwrite=true)
    interface = solver["LOWER_TO_UPPER"]
    interface.properties.dual_basis = false
    solver()
    node_ids, displacement = get_nodal_vector(interface.elements, "displacement", 0.0)
    node_ids, geometry = get_nodal_vector(interface.elements, "geometry", 0.0)
    u3 = [u[3] for u in displacement]
    maxabsu3 = maximum(abs(u3))
    stdabsu3 = std(abs(u3))
    info("max(abs(u3)) = $maxabsu3, std(abs(u3)) = $stdabsu3")
    @test isapprox(stdabsu3, 0.0; atol=1.0e-10)
end

@testset "small sliding contact patch test, tet10 + dual basis, alpha=0.2" begin
    solver = get_model(tet10_meshfile)
    solver.xdmf = Xdmf("contact_dl_quad_disp_results"; overwrite=true)
    interface = solver["LOWER_TO_UPPER"]
    interface.properties.dual_basis = true
    interface.properties.alpha = 0.2
    solver()
    node_ids, displacement = get_nodal_vector(interface.elements, "displacement", 0.0)
    node_ids, geometry = get_nodal_vector(interface.elements, "geometry", 0.0)
    u3 = [u[3] for u in displacement]
    maxabsu3 = maximum(abs(u3))
    stdabsu3 = std(abs(u3))
    info("max(abs(u3)) = $maxabsu3, std(abs(u3)) = $stdabsu3")
    @test isapprox(stdabsu3, 0.0; atol=1.0e-10)
end
