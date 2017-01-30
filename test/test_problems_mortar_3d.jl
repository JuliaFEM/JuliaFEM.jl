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
