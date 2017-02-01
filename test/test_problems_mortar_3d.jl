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

    #JuliaFEM.diagnose_interface(interface, 0.0)
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

#=
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
=#

#=
@testset "patch test temperature + abaqus inp + tet10, linear surface elements" begin
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

@testset "patch test temperature + abaqus inp + tet10, linear surface elements, splitting strategy" begin
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
    interface.properties.split_quadratic_slave_elements = true
    interface.properties.split_quadratic_master_elements = true

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
=#

