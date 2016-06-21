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
    p4.properties.dimension = 1
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
    interface.properties.dimension = 1

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
