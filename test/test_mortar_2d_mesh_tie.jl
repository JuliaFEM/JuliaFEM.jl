# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Postprocess
using JuliaFEM.Test

function get_test_model()
    X = Dict{Int64, Vector{Float64}}(
        1 => [0.0, 0.0],
        2 => [1.0, 0.0],
        3 => [1.0, 0.5],
        4 => [0.0, 0.5],
        5 => [0.0, 0.6],
        6 => [1.0, 0.6],
        7 => [1.0, 1.1],
        8 => [0.0, 1.1])
    el1 = Element(Quad4, [1, 2, 3, 4])
    el2 = Element(Quad4, [5, 6, 7, 8])
    el3 = Element(Seg2, [1, 2])
    el4 = Element(Seg2, [7, 8])
    el5 = Element(Seg2, [4, 3])
    el6 = Element(Seg2, [5, 6])
    update!([el1, el2, el3, el4, el5, el6], "geometry", X)
    update!([el1, el2], "youngs modulus", 96.0)
    update!([el1, el2], "poissons ratio", 1/3)
    update!([el3], "displacement 1", 0.0)
    update!([el3], "displacement 2", 0.0)
    update!([el4], "displacement 1", 0.0)
    update!([el4], "displacement 2", 0.0)
    update!(el6, "master elements", [el5])
    p1 = Problem(Elasticity, "body1", 2)
    p2 = Problem(Elasticity, "body2", 2)
    p3 = Problem(Dirichlet, "fixed", 2, "displacement")
    p4 = Problem(Mortar, "interface", 2, "displacement")
    push!(p1, el1)
    push!(p2, el2)
    push!(p3, el3, el4)
    push!(p4, el5, el6)
    return p1, p2, p3, p4
end

@testset "test adjust setting in 2d tie contact" begin
    p1, p2, p3, p4 = get_test_model()
    p1.properties.formulation = :plane_stress
    p2.properties.formulation = :plane_stress
    p4.properties.adjust = true
    p4.properties.rotate_normals = false
    solver = Solver(Nonlinear)
    push!(solver, p1, p2, p3, p4)
    call(solver)
    el5 = p4.elements[1]
    u = el5("displacement", [0.0], 0.0)
    info("u = $u")
    @test isapprox(u, [0.0, 0.05])
end

@testset "test that interface transfers constant field without error" begin
    meshfile = Pkg.dir("JuliaFEM") * "/test/testdata/block_2d.med"
    mesh = aster_read_mesh(meshfile)

    upper = Problem(Heat, "upper", 1)
    upper.elements = create_elements(mesh, "UPPER")
    update!(upper.elements, "temperature thermal conductivity", 1.0)

    lower = Problem(Heat, "lower", 1)
    lower.elements = create_elements(mesh, "LOWER")
    update!(lower.elements, "temperature thermal conductivity", 1.0)

    bc_upper = Problem(Dirichlet, "upper boundary", 1, "temperature")
    bc_upper.elements = create_elements(mesh, "UPPER_TOP")
    update!(bc_upper.elements, "temperature 1", 0.0)

    bc_lower = Problem(Dirichlet, "lower boundary", 1, "temperature")
    bc_lower.elements = create_elements(mesh, "LOWER_BOTTOM")
    update!(bc_lower.elements, "temperature 1", 1.0)

    interface = Problem(Mortar, "interface between upper and lower block", 1, "temperature")
    interface_slave_elements = create_elements(mesh, "LOWER_TOP")
    interface_master_elements = create_elements(mesh, "UPPER_BOTTOM")
    update!(interface_slave_elements, "master elements", interface_master_elements)
    interface.elements = [interface_master_elements; interface_slave_elements]

    solver = Solver()
    push!(solver, upper, lower, bc_upper, bc_lower, interface)
    call(solver)

    node_ids, temperature = get_nodal_vector(interface.elements, "temperature", 0.0)
    T = [t[1] for t in temperature]
    minT = minimum(T)
    maxT = maximum(T)
    info("minT = $minT, maxT = $maxT")
    @test isapprox(minT, 0.5)
    @test isapprox(maxT, 0.5)
end


#=
# TODO: if one forget plane_stress solver gives singular exception and it's
    hard to trace to the source of problem
@testset "expect clear error when trying to solve 2d model in 3d setting" begin
    p1, p2, p3, p4 = get_test_model()
#   p1.properties.formulation = :plane_stress
#   p2.properties.formulation = :plane_stress
    p4.properties.adjust = true
    p4.properties.rotate_normals = false
    solver = Solver(Nonlinear)
    solver.properties.linear_system_solver = :DirectLinearSolver_UMFPACK
    push!(solver, p1, p2, p3, p4)
    call(solver)
    el5 = p4.elements[1]
    u = el5("displacement", [0.0], 0.0)
    info("u = $u")
    @test isapprox(u, [0.0, 0.05])
end
=#

function JuliaFEM.get_mesh(::Type{Val{Symbol("1x1 block splitted to upper and lower")}})
    meshfile = Pkg.dir("JuliaFEM") * "/test/testdata/block_2d.med"
    mesh = aster_read_mesh(meshfile)
end

function JuliaFEM.get_model(::Type{Val{Symbol("splitted block, plane stress elasticity and mesh tie")}})
    mesh = get_mesh("1x1 block splitted to upper and lower")

    upper = Problem(Elasticity, "upper", 2)
    upper.properties.formulation = :plane_stress
    upper.elements = create_elements(mesh, "UPPER")
    update!(upper.elements, "youngs modulus", 100.0)
    update!(upper.elements, "poissons ratio", 1/3)

    lower = Problem(Elasticity, "lower", 2)
    lower.properties.formulation = :plane_stress
    lower.elements = create_elements(mesh, "LOWER")
    update!(lower.elements, "youngs modulus", 100.0)
    update!(lower.elements, "poissons ratio", 1/3)

    bc_upper = Problem(Dirichlet, "upper boundary", 2, "displacement")
    bc_upper.elements = create_elements(mesh, "UPPER_TOP")
#   update!(bc_upper.elements, "displacement 1", 0.1)
    update!(bc_upper.elements, "displacement 2", -0.1)

    bc_lower = Problem(Dirichlet, "lower boundary", 2, "displacement")
    bc_lower.elements = create_elements(mesh, "LOWER_BOTTOM")
#   update!(bc_lower.elements, "displacement 1", 0.0)
    update!(bc_lower.elements, "displacement 2", 0.0)

    bc_corner = Problem(Dirichlet, "fix model from lower left corner to prevent singularity", 2, "displacement")
    node_ids = find_nearest_nodes(mesh, [0.0, 0.0])
    bc_corner.elements = [Element(Poi1, node_ids)]
    update!(bc_corner.elements, "geometry", mesh.nodes)
    update!(bc_corner.elements, "displacement 1", 0.0)

    interface = Problem(Mortar, "interface between upper and lower block", 2, "displacement")
    interface_slave_elements = create_elements(mesh, "LOWER_TOP")
    interface_master_elements = create_elements(mesh, "UPPER_BOTTOM")
    update!(interface_slave_elements, "master elements", interface_master_elements)
    interface.elements = [interface_master_elements; interface_slave_elements]

    solver = Solver(Nonlinear)
    push!(solver, upper, lower, bc_upper, bc_lower, interface, bc_corner)

    return solver

end

@testset "test mesh tie with splitted block and plane stress elasticity" begin
    solver = get_model("splitted block, plane stress elasticity and mesh tie")
    upper, lower, bc_upper, bc_lower, interface = solver.problems
    call(solver)
    @test solver.properties.iteration == 2
    slave_elements = get_slave_elements(interface)
    node_ids, la = get_nodal_vector(slave_elements, "reaction force", 0.0)
    for lai in la
        @test isapprox(lai, [0.0, 10.0])
    end
end

function JuliaFEM.get_mesh(::Type{Val{Symbol("curved 2d block splitted to upper and lower")}})
    meshfile = Pkg.dir("JuliaFEM") * "/test/testdata/block_2d_curved.med"
    mesh = aster_read_mesh(meshfile)
end

function JuliaFEM.get_model(::Type{Val{Symbol("mesh tie with curved 2d block")}};
            dy=0.0, adjust=false, tolerance=0.0, rotate_normals=false, swap=false,
            dual_basis=false, use_forwarddiff=false)

    mesh = get_mesh("curved 2d block splitted to upper and lower")

    upper = Problem(Elasticity, "upper", 2)
    upper.properties.formulation = :plane_stress
    upper.elements = create_elements(mesh, "UPPER")
    update!(upper.elements, "youngs modulus", 96.0)
    update!(upper.elements, "poissons ratio", 1/3)

    lower = Problem(Elasticity, "lower", 2)
    lower.properties.formulation = :plane_stress
    lower.elements = create_elements(mesh, "LOWER")
    update!(lower.elements, "youngs modulus", 96.0)
    update!(lower.elements, "poissons ratio", 1/3)

    bc_upper = Problem(Dirichlet, "upper boundary", 2, "displacement")
    bc_upper.elements = create_elements(mesh, "UPPER_TOP")
    update!(bc_upper.elements, "displacement 1", 0.0)
    update!(bc_upper.elements, "displacement 2", dy)

    bc_lower = Problem(Dirichlet, "lower boundary", 2, "displacement")
    bc_lower.elements = create_elements(mesh, "LOWER_BOTTOM")
    update!(bc_lower.elements, "displacement 1", 0.0)
    update!(bc_lower.elements, "displacement 2", 0.0)

    interface = Problem(Mortar, "interface between upper and lower block", 2, "displacement")
    interface_slave_elements = create_elements(mesh, "LOWER_TOP")
    interface_master_elements = create_elements(mesh, "UPPER_BOTTOM")
    if swap
        interface_slave_elements, interface_master_elements = interface_master_elements, interface_slave_elements
    end
    update!(interface_slave_elements, "master elements", interface_master_elements)
    interface.elements = [interface_master_elements; interface_slave_elements]
    interface.properties.adjust = adjust
    interface.properties.distval = tolerance
    interface.properties.rotate_normals = rotate_normals
    interface.properties.dual_basis = dual_basis
    interface.properties.use_forwarddiff = use_forwarddiff

    solver = Solver(Nonlinear)
    push!(solver, upper, lower, bc_upper, bc_lower, interface)

    return solver

end

@testset "curved surface with adjust=true, standard lagrange, slave=lower surface, dy=0.0" begin
    # TODO: analytical solution now known, verify using other fem software
    solver = get_model("mesh tie with curved 2d block";
        adjust=true, tolerance=10, dy=0.0, rotate_normals=true,
        dual_basis=false)
    call(solver)
    interface = solver["interface between upper and lower block"]
    @test solver.properties.iteration == 2
    @test isapprox(norm(interface.assembly.u), 0.11339715157447851)
end

@testset "curved surface with adjust=true, dual lagrange, slave=lower surface, dy=0.0" begin
    # TODO: analytical solution now known, verify using other fem software
    solver = get_model("mesh tie with curved 2d block";
        adjust=true, tolerance=10, dy=0.0, rotate_normals=true,
        dual_basis=true)
    call(solver)
    interface = solver["interface between upper and lower block"]
    @test solver.properties.iteration == 2
    # differs -- why?
    @test isapprox(norm(interface.assembly.u), 0.11660422877751599)
end

@testset "curved surface with adjust=true, standard lagrange, slave=lower surface, dy=-0.1" begin
    # TODO: analytical solution now known, verify using other fem software
    solver = get_model("mesh tie with curved 2d block";
        adjust=true, tolerance=10, dy=-0.1, rotate_normals=true,
        dual_basis=false)
    call(solver)
    interface = solver["interface between upper and lower block"]
    @test solver.properties.iteration == 2
    @test isapprox(norm(interface.assembly.u), 0.34230262165505887)
end

@testset "curved surface, adjust=true, dual basis, slave=lower surface, dy=-0.1" begin
    # TODO: analytical solution now known, verify using other fem software
    solver = get_model("mesh tie with curved 2d block";
        adjust=true, tolerance=10, dy=-0.1, rotate_normals=true,
        dual_basis=true)
    call(solver)
    interface = solver["interface between upper and lower block"]
    @test solver.properties.iteration == 2
    @test isapprox(norm(interface.assembly.u), 0.34318800698017704)
end
