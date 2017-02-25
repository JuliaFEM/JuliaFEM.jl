# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Postprocess
using JuliaFEM.Testing
using JuliaFEM.Abaqus: create_surface_elements

tet4_meshfile = "test_problems_mortar_3d/tet4.inp"
tet10_meshfile = "test_problems_mortar_3d/tet10.inp"

@testset "patch test displacement + abaqus inp + tet4 + adjust" begin

    mesh = abaqus_read_mesh(tet4_meshfile)
    
    upper = Problem(Elasticity, "UPPER", 3)
    upper.elements = create_elements(mesh, "UPPER")
    update!(upper, "youngs modulus", 288.0)
    update!(upper, "poissons ratio", 1/3)

    lower = Problem(Elasticity, "LOWER", 3)
    lower.elements = create_elements(mesh, "LOWER")
    update!(lower, "youngs modulus", 288.0)
    update!(lower, "poissons ratio", 1/3)

    bc_upper = Problem(Dirichlet, "UPPER_TOP", 3, "displacement")
    bc_upper.elements = create_surface_elements(mesh, "UPPER_TOP")
    update!(bc_upper, "displacement 3", -0.2)

    bc_lower = Problem(Dirichlet, "LOWER_BOTTOM", 3, "displacement")
    bc_lower.elements = create_surface_elements(mesh, "LOWER_BOTTOM")
    update!(bc_lower, "displacement 3", 0.0)

    bc_sym13 = Problem(Dirichlet, "SYM13", 3, "displacement")
    bc_sym13.elements = [create_surface_elements(mesh, "LOWER_SYM13"); create_surface_elements(mesh, "UPPER_SYM13")]
    update!(bc_sym13, "displacement 2", 0.0)

    bc_sym23 = Problem(Dirichlet, "SYM23", 3, "displacement")
    bc_sym23.elements = [create_surface_elements(mesh, "LOWER_SYM23"); create_surface_elements(mesh, "UPPER_SYM23")]
    update!(bc_sym23, "displacement 1", 0.0)

    interface = Problem(Contact, "LOWER_TO_UPPER", 3, "displacement")
    interface_slave_elements = create_surface_elements(mesh, "LOWER_TO_UPPER")
    interface_master_elements = create_surface_elements(mesh, "UPPER_TO_LOWER")
    update!(interface_slave_elements, "master elements", interface_master_elements)
    interface.elements = [interface_slave_elements; interface_master_elements]

    append!(interface.assembly.removed_dofs, [1316, 1319, 1358, 1387, 1388, 1492, 1597, 1627])

    interface.properties.linear_surface_elements = false
    interface.properties.split_quadratic_slave_elements = false
    interface.properties.split_quadratic_master_elements = false
    interface.properties.adjust = true
    interface.properties.dual_basis = false

    solver = LinearSolver(upper, lower, bc_upper, bc_lower, bc_sym13, bc_sym23, interface)
    solver.xdmf = Xdmf("contact_sl_lin_disp_results")
    solver()

    node_ids, displacement = get_nodal_vector(interface.elements, "displacement", 0.0)
    node_ids, geometry = get_nodal_vector(interface.elements, "geometry", 0.0)
    u3 = [u[3] for u in displacement]
    maxabsu3 = maximum(abs(u3))
    stdabsu3 = std(abs(u3))
    info("tet10 block: max(abs(u3)) = $maxabsu3, std(abs(u3)) = $stdabsu3")
    @test isapprox(stdabsu3, 0.0; atol=1.0e-10)
end
