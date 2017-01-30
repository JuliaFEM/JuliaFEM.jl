# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Postprocess
using JuliaFEM.Testing
using JuliaFEM.Abaqus: create_surface_elements

@testset "test that interface transfers constant field without error" begin
    meshfile = Pkg.dir("JuliaFEM") * "/test/testdata/block_3d.med"
    mesh = aster_read_mesh(meshfile)

    upper = Problem(Heat, "upper", 1)
    upper.elements = create_elements(mesh, "UPPER")
    update!(upper, "temperature thermal conductivity", 1.0)
    lower = Problem(Heat, "lower", 1)
    lower.elements = create_elements(mesh, "LOWER")
    update!(lower, "temperature thermal conductivity", 1.0)

    bc_upper = Problem(Dirichlet, "upper boundary", 1, "temperature")
    bc_upper.elements = create_elements(mesh, "UPPER_TOP")
    update!(bc_upper, "temperature 1", 0.0)

    bc_lower = Problem(Dirichlet, "lower boundary", 1, "temperature")
    bc_lower.elements = create_elements(mesh, "LOWER_BOTTOM")
    update!(bc_lower, "temperature 1", 1.0)

    interface = Problem(Mortar, "interface between upper and lower block", 1, "temperature")
    interface_slave_elements = create_elements(mesh, "LOWER_TOP")
    interface_master_elements = create_elements(mesh, "UPPER_BOTTOM")
    update!(interface_slave_elements, "master elements", interface_master_elements)
    interface.elements = [interface_master_elements; interface_slave_elements]

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

@testset "patch test temperature + abaqus inp + tet10" begin
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

    interface = Problem(Mortar, "LOWER_TO_UPPER", 1, "temperature")
    interface_slave_elements = create_surface_elements(mesh, "LOWER_TO_UPPER")
    interface_master_elements = create_surface_elements(mesh, "UPPER_TO_LOWER")
    update!(interface_slave_elements, "master elements", interface_master_elements)
    interface.elements = [interface_slave_elements; interface_master_elements]

    interface.properties.linear_surface_elements = true
    interface.properties.split_quadratic_slave_elements = false
    interface.properties.split_quadratic_master_elements = false

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

@testset "patch test displacement + abaqus inp + tet10" begin
    meshfile = Pkg.dir("JuliaFEM") * "/test/testdata/test_problems_mortar_3d_tet10.inp"
    mesh = abaqus_read_mesh(meshfile)

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

    interface = Problem(Mortar, "LOWER_TO_UPPER", 3, "displacement")
    interface_slave_elements = create_surface_elements(mesh, "LOWER_TO_UPPER")
    interface_master_elements = create_surface_elements(mesh, "UPPER_TO_LOWER")
    update!(interface_slave_elements, "master elements", interface_master_elements)
    interface.elements = [interface_slave_elements; interface_master_elements]

    removed_dofs = [1316, 1319, 1388, 1361, 1358, 1391, 1325, 1600, 1492, 1657, 1627, 1387, 1630, 1597]
    append!(interface.assembly.removed_dofs, removed_dofs)

    interface.properties.linear_surface_elements = false
    interface.properties.split_quadratic_slave_elements = false
    interface.properties.split_quadratic_master_elements = false

    solver = LinearSolver(upper, lower, bc_upper, bc_lower, bc_sym13, bc_sym23, interface)
    solver()
    node_ids, displacement = get_nodal_vector(interface.elements, "displacement", 0.0)
    node_ids, geometry = get_nodal_vector(interface.elements, "geometry", 0.0)
    u1 = [u[1] for u in displacement]
    u2 = [u[2] for u in displacement]
    u3 = [u[3] for u in displacement]
    X3 = [X[3] for X in geometry]
    minu3 = minimum(u3)
    maxu3 = maximum(u3)
    stdu3 = std(u3)
    minX3 = minimum(X3)
    maxX3 = maximum(X3)
    stdX3 = std(X3)
    info("tet10 block: minimum u3 = $minu3, maximum u3 = $maxu3, stdu3 = $stdu3")
    info("tet10 block: minimum X3 = $minX3, maximum X3 = $maxX3, stdX3 = $stdX3")
    @test isapprox(minu3, -0.1)
    @test isapprox(maxu3, -0.1)
end

