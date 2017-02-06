# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Postprocess
using JuliaFEM.Testing
using JuliaFEM.Abaqus: create_surface_elements

@testset "patch test temperature + abaqus inp + tet4" begin
    meshfile = Pkg.dir("JuliaFEM") * "/test/testdata/test_problems_mortar_3d_tet4.inp"
    mesh = abaqus_read_mesh(meshfile)

    upper = Problem(Heat, "UPPER", 1)
    upper.elements = create_elements(mesh, "UPPER")
    update!(upper, "temperature thermal conductivity", 1.0)

    lower = Problem(Heat, "LOWER", 1)
    lower.elements = create_elements(mesh, "LOWER")
    update!(lower, "temperature thermal conductivity", 1.0)

    bc_upper = Problem(Dirichlet, "UPPER_TOP", 1, "temperature")
    bc_upper.elements = create_surface_elements(mesh, "UPPER_TOP")
    update!(bc_upper, "temperature 1", 0.0)

    bc_lower = Problem(Dirichlet, "LOWER_BOTTOM", 1, "temperature")
    bc_lower.elements = create_surface_elements(mesh, "LOWER_BOTTOM")
    update!(bc_lower, "temperature 1", 1.0)

    interface = Problem(Mortar, "interface between upper and lower block", 1, "temperature")
    interface_slave_elements = create_surface_elements(mesh, "LOWER_TO_UPPER")
    interface_master_elements = create_surface_elements(mesh, "UPPER_TO_LOWER")
    update!(interface_slave_elements, "master elements", interface_master_elements)
    interface.elements = [interface_master_elements; interface_slave_elements]

    JuliaFEM.diagnose_interface(interface, 0.0)
    solver = LinearSolver(upper, lower, bc_upper, bc_lower, interface)

    solver()
    
    node_ids, temperature = get_nodal_vector(interface.elements, "temperature", 0.0)
    T = [t[1] for t in temperature]
    minT = minimum(T)
    maxT = maximum(T)
    info("minT = $minT, maxT = $maxT")
    @test isapprox(minT, 0.5)
    @test isapprox(maxT, 0.5)

    #=
    initialize!(solver)
    assemble!(solver)
    M, K, Kg, f, fg = get_field_assembly(solver)
    Kb, C1, C2, D, fb, g = get_boundary_assembly(solver)
    K = K + Kg + Kb
    f = f + fg + fb
    K = 1/2*(K + K')
    M = 1/2*(M + M')
    =#

end

@testset "patch test temperature + abaqus inp + tet4 + dual basis + adjust" begin
    meshfile = Pkg.dir("JuliaFEM") * "/test/testdata/test_problems_mortar_3d_tet4.inp"
    mesh = abaqus_read_mesh(meshfile)

    upper = Problem(Heat, "UPPER", 1)
    upper.elements = create_elements(mesh, "UPPER")
    update!(upper, "temperature thermal conductivity", 1.0)

    lower = Problem(Heat, "LOWER", 1)
    lower.elements = create_elements(mesh, "LOWER")
    update!(lower, "temperature thermal conductivity", 1.0)

    bc_upper = Problem(Dirichlet, "UPPER_TOP", 1, "temperature")
    bc_upper.elements = create_surface_elements(mesh, "UPPER_TOP")
    update!(bc_upper, "temperature 1", 0.0)

    bc_lower = Problem(Dirichlet, "LOWER_BOTTOM", 1, "temperature")
    bc_lower.elements = create_surface_elements(mesh, "LOWER_BOTTOM")
    update!(bc_lower, "temperature 1", 1.0)

    interface = Problem(Mortar, "interface between upper and lower block", 1, "temperature")
    interface_slave_elements = create_surface_elements(mesh, "LOWER_TO_UPPER")
    interface_master_elements = create_surface_elements(mesh, "UPPER_TO_LOWER")
    update!(interface_slave_elements, "master elements", interface_master_elements)
    interface.elements = [interface_master_elements; interface_slave_elements]
    interface.properties.dual_basis = true
    #interface.properties.adjust = true

    JuliaFEM.diagnose_interface(interface, 0.0)
    solver = LinearSolver(upper, lower, bc_upper, bc_lower, interface)

    solver()
    
    node_ids, temperature = get_nodal_vector(interface.elements, "temperature", 0.0)
    T = [t[1] for t in temperature]
    minT = minimum(T)
    maxT = maximum(T)
    info("minT = $minT, maxT = $maxT")
    @test isapprox(minT, 0.5)
    @test isapprox(maxT, 0.5)

end

@testset "patch test temperature + abaqus inp + tet10, quadratic surface elements" begin
    meshfile = Pkg.dir("JuliaFEM") * "/test/testdata/test_problems_mortar_3d_tet10.inp"
    mesh = abaqus_read_mesh(meshfile)

    upper = Problem(Heat, "UPPER", 1)
    upper.elements = create_elements(mesh, "UPPER")
    update!(upper, "temperature thermal conductivity", 1.0)

    lower = Problem(Heat, "LOWER", 1)
    lower.elements = create_elements(mesh, "LOWER")
    update!(lower, "temperature thermal conductivity", 1.0)

    bc_upper = Problem(Dirichlet, "UPPER_TOP", 1, "temperature")
    bc_upper.elements = create_surface_elements(mesh, "UPPER_TOP")
    update!(bc_upper, "temperature 1", 0.0)

    bc_lower = Problem(Dirichlet, "LOWER_BOTTOM", 1, "temperature")
    bc_lower.elements = create_surface_elements(mesh, "LOWER_BOTTOM")
    update!(bc_lower, "temperature 1", 1.0)

    interface = Problem(Mortar, "interface between upper and lower block", 1, "temperature")
    interface_slave_elements = create_surface_elements(mesh, "LOWER_TO_UPPER")
    interface_master_elements = create_surface_elements(mesh, "UPPER_TO_LOWER")
    update!(interface_slave_elements, "master elements", interface_master_elements)
    interface.elements = [interface_master_elements; interface_slave_elements]
    
    interface.properties.linear_surface_elements = false
    interface.properties.split_quadratic_slave_elements = false
    interface.properties.split_quadratic_master_elements = false
    interface.properties.alpha = 0.0
#   JuliaFEM.diagnose_interface(interface, 0.0)

    solver = LinearSolver(upper, lower, bc_upper, bc_lower, interface)
    solver()
    
    node_ids, temperature = get_nodal_vector(interface.elements, "temperature", 0.0)
    T = [t[1] for t in temperature]
    
    minT = minimum(T)
    maxT = maximum(T)
    stdT = std(T)
    info("minT = $minT, maxT = $maxT, stdT = $stdT")
    @test maxT - minT < 5.0e-4
    @test isapprox(stdT, 0.0; atol=1.0e-4)

end

@testset "patch test displacement + abaqus inp + tet4 + adjust + dual basis" begin
    meshfile = Pkg.dir("JuliaFEM") * "/test/testdata/test_problems_mortar_3d_tet4.inp"

    mesh = abaqus_read_mesh(meshfile)
    # modify mesh a bit, find all nodes in elements in element set UPPER and put 0.2 to X3 to test adjust
    JuliaFEM.Preprocess.create_node_set_from_element_set!(mesh, :UPPER)
    for nid in mesh.node_sets[:UPPER]
        mesh.nodes[nid][3] += 0.2
    end

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
    update!(bc_upper, "displacement 3", 0.0)

    bc_lower = Problem(Dirichlet, "LOWER_BOTTOM", 3, "displacement")
    bc_lower.elements = create_surface_elements(mesh, "LOWER_BOTTOM")
    update!(bc_lower, "displacement 3", 0.0)

    bc_sym13 = Problem(Dirichlet, "SYM13", 3, "displacement")
    bc_sym13.elements = [create_surface_elements(mesh, "LOWER_SYM13"); create_surface_elements(mesh, "UPPER_SYM13")]
    update!(bc_sym13, "displacement 2", 0.0)

    bc_sym23 = Problem(Dirichlet, "SYM23", 3, "displacement")
    bc_sym23.elements = [create_surface_elements(mesh, "LOWER_SYM23"); create_surface_elements(mesh, "UPPER_SYM23")]
    update!(bc_sym23, "displacement 1", 0.0)

    interface = Problem(Mortar, "LOWER_TO_UPPER", 3, "displacement")
    interface_slave_elements = create_surface_elements(mesh, "LOWER_TO_UPPER")
    interface_master_elements = create_surface_elements(mesh, "UPPER_TO_LOWER")
    update!(interface_slave_elements, "master elements", interface_master_elements)
    interface.elements = [interface_slave_elements; interface_master_elements]

    append!(interface.assembly.removed_dofs, [1316, 1319, 1358, 1387, 1388, 1492, 1597, 1627])

    interface.properties.linear_surface_elements = false
    interface.properties.split_quadratic_slave_elements = false
    interface.properties.split_quadratic_master_elements = false
    interface.properties.adjust = true
    interface.properties.dual_basis = true

    solver = LinearSolver(upper, lower, bc_upper, bc_lower, bc_sym13, bc_sym23, interface)
    #solver.xdmf = Xdmf("results")
    solver()

    node_ids, displacement = get_nodal_vector(interface.elements, "displacement", 0.0)
    node_ids, geometry = get_nodal_vector(interface.elements, "geometry", 0.0)
    u3 = [u[3] for u in displacement]
    maxabsu3 = maximum(abs(u3))
    stdabsu3 = std(abs(u3))
    info("tet10 block: max(abs(u3)) = $maxabsu3, std(abs(u3)) = $stdabsu3")
    @test isapprox(stdabsu3, 0.0; atol=1.0e-12)
end

@testset "patch test displacement + abaqus inp + tet10 + adjust + dual basis + alpha=0.2" begin
    meshfile = Pkg.dir("JuliaFEM") * "/test/testdata/test_problems_mortar_3d_tet10.inp"

    mesh = abaqus_read_mesh(meshfile)
    # modify mesh a bit, find all nodes in elements in element set UPPER and put 0.2 to X3 to test adjust
    JuliaFEM.Preprocess.create_node_set_from_element_set!(mesh, :UPPER)
    for nid in mesh.node_sets[:UPPER]
        mesh.nodes[nid][3] += 0.2
    end

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
    update!(bc_upper, "displacement 3", 0.0)

    bc_lower = Problem(Dirichlet, "LOWER_BOTTOM", 3, "displacement")
    bc_lower.elements = create_surface_elements(mesh, "LOWER_BOTTOM")
    update!(bc_lower, "displacement 3", 0.0)

    bc_sym13 = Problem(Dirichlet, "SYM13", 3, "displacement")
    bc_sym13.elements = [create_surface_elements(mesh, "LOWER_SYM13"); create_surface_elements(mesh, "UPPER_SYM13")]
    update!(bc_sym13, "displacement 2", 0.0)

    bc_sym23 = Problem(Dirichlet, "SYM23", 3, "displacement")
    bc_sym23.elements = [create_surface_elements(mesh, "LOWER_SYM23"); create_surface_elements(mesh, "UPPER_SYM23")]
    update!(bc_sym23, "displacement 1", 0.0)

    interface = Problem(Mortar, "LOWER_TO_UPPER", 3, "displacement")
    interface_slave_elements = create_surface_elements(mesh, "LOWER_TO_UPPER")
    interface_master_elements = create_surface_elements(mesh, "UPPER_TO_LOWER")
    update!(interface_slave_elements, "master elements", interface_master_elements)
    interface.elements = [interface_slave_elements; interface_master_elements]

    removed_dofs = [1316, 1319, 1325, 1358, 1361, 1387, 1388, 1391, 1492, 1597, 1600, 1627, 1630, 1657]
    append!(interface.assembly.removed_dofs, removed_dofs)

    interface.properties.linear_surface_elements = false
    interface.properties.split_quadratic_slave_elements = false
    interface.properties.split_quadratic_master_elements = false
    interface.properties.adjust = true
    interface.properties.dual_basis = true
    interface.properties.alpha = 0.2

    solver = LinearSolver(upper, lower, bc_upper, bc_lower, bc_sym13, bc_sym23, interface)
    #solver.xdmf = Xdmf("results")
    solver()

    node_ids, displacement = get_nodal_vector(interface.elements, "displacement", 0.0)
    node_ids, geometry = get_nodal_vector(interface.elements, "geometry", 0.0)
    u3 = [u[3] for u in displacement]
    maxabsu3 = maximum(abs(u3))
    stdabsu3 = std(abs(u3))
    info("tet10 block: max(abs(u3)) = $maxabsu3, std(abs(u3)) = $stdabsu3")
    @test isapprox(stdabsu3, 0.0; atol=3.0e-5)
end

@testset "patch test temperature + abaqus inp + tet10 + quadratic surface elements + dual basis + alpha=0.2" begin
    meshfile = Pkg.dir("JuliaFEM") * "/test/testdata/test_problems_mortar_3d_tet10.inp"
    mesh = abaqus_read_mesh(meshfile)

    upper = Problem(Heat, "UPPER", 1)
    upper.elements = create_elements(mesh, "UPPER")
    update!(upper, "temperature thermal conductivity", 1.0)

    lower = Problem(Heat, "LOWER", 1)
    lower.elements = create_elements(mesh, "LOWER")
    update!(lower, "temperature thermal conductivity", 1.0)

    bc_upper = Problem(Dirichlet, "UPPER_TOP", 1, "temperature")
    bc_upper.elements = create_surface_elements(mesh, "UPPER_TOP")
    update!(bc_upper, "temperature 1", 0.0)

    bc_lower = Problem(Dirichlet, "LOWER_BOTTOM", 1, "temperature")
    bc_lower.elements = create_surface_elements(mesh, "LOWER_BOTTOM")
    update!(bc_lower, "temperature 1", 1.0)

    interface = Problem(Mortar, "interface between upper and lower block", 1, "temperature")
    interface_slave_elements = create_surface_elements(mesh, "LOWER_TO_UPPER")
    interface_master_elements = create_surface_elements(mesh, "UPPER_TO_LOWER")
    update!(interface_slave_elements, "master elements", interface_master_elements)
    interface.elements = [interface_master_elements; interface_slave_elements]
    
    interface.properties.linear_surface_elements = false
    interface.properties.split_quadratic_slave_elements = false
    interface.properties.split_quadratic_master_elements = false
    interface.properties.dual_basis = true
    interface.properties.alpha = 0.2

    solver = LinearSolver(upper, lower, bc_upper, bc_lower, interface)
    solver()
    
    node_ids, temperature = get_nodal_vector(interface.elements, "temperature", 0.0)
    #node_ids, temperature = get_nodal_vector(interface_slave_elements, "temperature", 0.0)
    #=
    for (j, (nid, T)) in enumerate(zip(node_ids, temperature))
        info("$j: $nid -> $(T[1])")
        j == 10 && break
    end
    =#
    T = [t[1] for t in temperature]
    minT = minimum(T)
    maxT = maximum(T)
    stdT = std(T)
    info("minT = $minT, maxT = $maxT, stdT = $stdT")
    @test maxT - minT < 5.0e-4
    @test isapprox(stdT, 0.0; atol=1.0e-4)
end
