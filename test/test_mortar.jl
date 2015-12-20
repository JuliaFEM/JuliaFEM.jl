# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

module MortarTests

using JuliaFEM.Test

using JuliaFEM.Core: Element, Seg2, Quad4, Tri3, Hex8, MortarProblem, Assembly, assemble!,
                     get_connectivity, update!
using JuliaFEM.Core: PlaneStressElasticityProblem, DirichletProblem, DirectSolver

# 2d stuff
using JuliaFEM.Core: project_from_slave_to_master, project_from_master_to_slave

# 3d stuff
using JuliaFEM.Core: create_auxiliary_plane, project_point_to_auxiliary_plane,
                     get_edge_intersections, get_points_inside_triangle,
                     clip_polygon, calculate_polygon_centerpoint,
                     project_point_from_plane_to_surface, assemble,
                     calculate_normal_tangential_coordinates!,
                     is_point_inside_convex_polygon

using JuliaFEM.Core: LinearElasticityProblem

function get_test_2d_model()
    # this is hand calculated and given as an example in my thesis
    N = Vector[
        [0.0, 2.0], [1.0, 2.0], [2.0, 2.0],
        [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
        [0.0, 1.0], [5/4, 1.0], [2.0, 1.0],
        [0.0, 1.0], [3/4, 1.0], [2.0, 1.0]]
    rotation_matrix(phi) = [cos(phi) -sin(phi); sin(phi) cos(phi)]

    master1 = Seg2([7, 8])
    master1["geometry"] = Vector[N[7], N[8]]
    master2 = Seg2([8, 9])
    master2["geometry"] = Vector[N[8], N[9]]

#=
    master1 = Seg2([9, 8])
    master1["geometry"] = Vector[N[9], N[8]]
    master2 = Seg2([8, 7])
    master2["geometry"] = Vector[N[8], N[7]]
=#

    slave1 = Seg2([10, 11])
    slave1["geometry"] = Vector[N[10], N[11]]
    # should be n = [0 -1]' and t = [1 0]'
    slave1["normal-tangential coordinates"] = Matrix[rotation_matrix(-pi/2), rotation_matrix(-pi/2)]
    slave1["master elements"] = Element[master1, master2]

    slave2 = Seg2([11, 12])
    slave2["geometry"] = Vector[N[11], N[12]]
    # should be n = [0 -1]' and t = [1 0]'
    slave2["normal-tangential coordinates"] = Matrix[rotation_matrix(-pi/2), rotation_matrix(-pi/2)]
    slave2["master elements"] = Element[master1, master2]

    return [slave1, slave2], [master1, master2]
end


function test_calc_flat_2d_projection_slave_to_master()
    slaves, masters = get_test_2d_model()
    slave1, slave2 = slaves
    master1, master2 = masters

    xi2a = project_from_slave_to_master(slave1, master1, [-1.0])
    @test xi2a == [-1.0]

    xi2b = project_from_slave_to_master(slave1, master1, [1.0])
    @test xi2b == [ 0.2]
    X2 = master1("geometry", xi2b, 0.0)
    @test X2 == [3/4, 1.0]
end


function test_calc_flat_2d_projection_master_to_slave()
    slaves, masters = get_test_2d_model()
    slave1, slave2 = slaves
    master1, master2 = masters
    xi1a = project_from_master_to_slave(slave1, master1, [-1.0])
    @test xi1a == [-1.0]
    xi1b = project_from_master_to_slave(slave1, master1, [1.0])
    X1 = slave1("geometry", xi1b, 0.0)
    @test X1 == [5/4, 1.0]
end
#test_calc_flat_2d_projection_master_to_slave()


function test_calc_flat_2d_projection_rotated()
    master1 = Seg2([3, 4])
    master1["geometry"] = Vector{Float64}[[0.0, 1.0], [0.0, 0.0]]
    slave1 = Seg2([1, 2])
    slave1["geometry"] = Vector{Float64}[[0.0, 0.0], [0.0, 1.0]]
    slave1["normal-tangential coordinates"] = Matrix{Float64}[[1.0 0.0; 0.0 1.0], [1.0 0.0; 0.0 1.0]]
    xi = project_from_master_to_slave(slave1, master1, [-1.0])
    info("xi = $xi")
    @test xi == [ 1.0]
    xi = project_from_master_to_slave(slave1, master1, [1.0])
    info("xi = $xi")
    @test xi == [-1.0]

    xi = project_from_slave_to_master(slave1, master1, [-1.0])
    info("xi = $xi")
    @test xi == [ 1.0]
    xi = project_from_slave_to_master(slave1, master1, [1.0])
    info("xi = $xi")
    @test xi == [-1.0]

end


function test_create_flat_2d_assembly()
    slaves, masters = get_test_2d_model()
    slave1, slave2 = slaves
    master1, master2 = masters

    info("creating problem")
    problem = MortarProblem("temperature", 1)
    info("pushing slave elements to problem")
    push!(problem, slave1)
    push!(problem, slave2)

    B_expected = zeros(12, 12)
    S1 = [10, 11]
    M1 = [7, 8]
    B_expected[S1,S1] += [1/4 1/8; 1/8 1/4]
    B_expected[S1,M1] -= [3/10 3/40; 9/40 3/20]

    info("creating assembly")
    assembly = Assembly()
    assemble!(assembly, problem, slave1, 0.0)
    B = round(full(assembly.stiffness_matrix, 12, 12), 6)
    info("size of B = $(size(B))")
    info("B matrix in first slave element = \n$(B[10:11,:])")
    info("B matrix expected = \n$(B_expected[10:11,:])")
    @test isapprox(B, B_expected)

    fill!(B_expected, 0.0)
    empty!(assembly)

    S2 = [11, 12]
    M2 = [7, 8]
    B_expected[S2,S2] += [49/150 11/150; 11/150 2/75]
    B_expected[S2,M2] -= [13/150 47/150; 1/75 13/150]
    S3 = [11, 12]
    M3 = [8, 9]
    B_expected[S3,S3] += [9/100 27/200; 27/200 39/100]
    B_expected[S3,M3] -= [3/20 3/40; 9/40 3/10]
    assemble!(assembly, problem, slave2, 0.0)
    B = full(assembly.stiffness_matrix)
    info("size of B = $(size(B))")
    info("B matrix in second slave element = \n$(B[11:12,:])")
    info("B matrix expected = \n$(B_expected[11:12,:])")

    @test isapprox(B, B_expected)
end
#test_create_flat_2d_assembly()


function test_2d_mortar_multiple_bodies_multiple_dirichlet_bc()
    N = Vector[
        [0.0, 0.0], [1.0, 0.0],
        [0.0, 1.0], [1.0, 1.0],
        [0.0, 1.0], [1.0, 1.0],
        [0.0, 2.0], [1.0, 2.0]]

    e1 = Quad4([1, 2, 4, 3])
    e1["geometry"] = Vector[N[1], N[2], N[4], N[3]]
    e2 = Quad4([5, 6, 8, 7])
    e2["geometry"] = Vector[N[5], N[6], N[8], N[7]]
    for el in [e1, e2]
        el["youngs modulus"] = 900.0
        el["poissons ratio"] = 0.25
    end
    b1 = Seg2([7, 8])
    b1["geometry"] = Vector[N[7], N[8]]
    b1["displacement traction force"] = Vector[[0.0, -100.0], [0.0, -100.0]]

    body1 = PlaneStressElasticityProblem()
    push!(body1, e1)

    body2 = PlaneStressElasticityProblem()
    push!(body2, e2)
    push!(body2, b1)

    # boundary elements for dirichlet dx=0
    dx1 = Seg2([1, 3])
    dx1["geometry"] = Vector[N[1], N[3]]
    dx2 = Seg2([5, 7])
    dx2["geometry"] = Vector[N[5], N[7]]
    for dx in [dx1, dx2]
        dx["displacement 1"] = 0.0
    end

    boundary1 = DirichletProblem("displacement", 2)
    push!(boundary1, dx1)
    push!(boundary1, dx2)

    # boundary elements for dirichlet dy=0
    dy1 = Seg2([1, 2])
    dy1["geometry"] = Vector[N[1], N[2]]
    dy1["displacement 2"] = 0.0

    boundary2 = DirichletProblem("displacement", 2)
    push!(boundary2, dy1)

    # mortar boundary between two bodies
    rotation_matrix(phi) = [cos(phi) -sin(phi); sin(phi) cos(phi)]

    master1 = Seg2([3, 4])
    master1["geometry"] = Vector[N[3], N[4]]

    slave1 = Seg2([5, 6])
    slave1["geometry"] = Vector[N[5], N[6]]
    slave1["normal-tangential coordinates"] = Matrix[rotation_matrix(-pi/2), rotation_matrix(-pi/2)]
    slave1["master elements"] = Element[master1]

    boundary3 = MortarProblem("displacement", 2)
    push!(boundary3, slave1)

    solver = DirectSolver()
    push!(solver, body1)
    push!(solver, body2)
    push!(solver, boundary1)
    push!(solver, boundary2)
    push!(solver, boundary3)

    solver.name = "test_2d_mortar_multiple_bodies_multiple_dirichlet_bcs"
    solver.dump_matrices = true
    solver.method = :UMFPACK
    # launch solver
    solver(0.0)

    disp = e2("displacement", [1.0, 1.0], 0.0)
    info("displacement at tip: $disp")
    # code aster verification, two_elements.comm
    @test isapprox(disp, [3.17431158889468E-02, -2.77183037855653E-01])
end
#test_2d_mortar_multiple_bodies_multiple_dirichlet_bc()


function test_2d_mortar_three_bodies_shared_nodes()
    N = Dict{Int, Vector{Float64}}(
        1 => [0.0, 0.0],
        2 => [2.0, 0.0],
        3 => [0.0, 1.0],
        4 => [2.0, 1.0],
        5 => [0.0, 1.0],
        6 => [1.3, 1.0],
        7 => [0.0, 2.0],
        8 => [1.3, 2.0],
        9 => [1.3, 1.0],
        10 => [2.0, 1.0],
        11 => [1.3, 2.0],
        12 => [2.0, 2.0])

    e1 = Quad4([1, 2, 4, 3])
    e1["geometry"] = Vector[N[1], N[2], N[4], N[3]]

    e2 = Quad4([5, 6, 8, 7])
    e2["geometry"] = Vector[N[5], N[6], N[8], N[7]]

    e3 = Quad4([9, 10, 12, 11])
    e3["geometry"] = Vector[N[9], N[10], N[12], N[11]]

    for el in [e1, e2, e3]
        el["youngs modulus"] = 900.0
        el["poissons ratio"] = 0.25
    end

    b1 = Seg2([7, 8])
    b1["geometry"] = Vector[N[7], N[8]]
    b1["displacement traction force"] = Vector[[0.0, -100.0], [0.0, -100.0]]

    b2 = Seg2([11, 12])
    b2["geometry"] = Vector[N[11], N[12]]
    b2["displacement traction force"] = Vector[[0.0, -100.0], [0.0, -100.0]]

    body1 = PlaneStressElasticityProblem()
    push!(body1, e1)

    body2 = PlaneStressElasticityProblem()
    push!(body2, e2)
    push!(body2, b1)

    body3 = PlaneStressElasticityProblem()
    push!(body3, e3)
    push!(body3, b2)

    # boundary elements for dirichlet dx=0
    dx1 = Seg2([1, 3])
    dx1["geometry"] = Vector[N[1], N[3]]
    dx2 = Seg2([5, 7])
    dx2["geometry"] = Vector[N[5], N[7]]
    for dx in [dx1, dx2]
        dx["displacement 1"] = 0.0
    end

    bc1 = DirichletProblem("displacement", 2)
    push!(bc1, dx1)
    push!(bc1, dx2)

    # boundary elements for dirichlet dy=0
    dy1 = Seg2([1, 2])
    dy1["geometry"] = Vector[N[1], N[2]]
    dy1["displacement 2"] = 0.0

    bc2 = DirichletProblem("displacement", 2)
    push!(bc2, dy1)

    # mortar boundary between body 1 and body 2
    rotation_matrix(phi) = [cos(phi) -sin(phi); sin(phi) cos(phi)]

    master1 = Seg2([3, 4])
    master1["geometry"] = Vector[N[3], N[4]]

    slave1 = Seg2([5, 6])
    slave1["geometry"] = Vector[N[5], N[6]]
    slave1["normal-tangential coordinates"] = Matrix[rotation_matrix(-pi/2), rotation_matrix(-pi/2)]
    slave1["master elements"] = Element[master1]
    bc3 = MortarProblem("displacement", 2)
    push!(bc3, slave1)

    # mortar boundary between body 1 and body 3
    slave2 = Seg2([9, 10])
    slave2["geometry"] = Vector[N[9], N[10]]
    slave2["normal-tangential coordinates"] = Matrix[rotation_matrix(-pi/2), rotation_matrix(-pi/2)]
    slave2["master elements"] = Element[master1]
    bc4 = MortarProblem("displacement", 2)
    push!(bc4, slave2)

    # mortar boundary between body 2 and body 3
    master2 = Seg2([9, 11])
    master2["geometry"] = Vector[N[9], N[11]]

    slave3 = Seg2([6, 8])
    slave3["geometry"] = Vector[N[6], N[8]]
    #slave3["normal-tangential coordinates"] = Matrix[rotation_matrix(-pi/2), rotation_matrix(-pi/2)]
    slave3["normal-tangential coordinates"] = Matrix[rotation_matrix(0.0), rotation_matrix(0.0)]
    slave3["master elements"] = Element[master2]
    bc5 = MortarProblem("displacement", 2)
    push!(bc5, slave3)

    solver = DirectSolver()
    push!(solver, body1)
    push!(solver, body2)
    push!(solver, body3)

    push!(solver, bc1)
    push!(solver, bc2)

    push!(solver, bc3)
    push!(solver, bc4)
    push!(solver, bc5)

    # launch solver
    solver.method = :UMFPACK
    solver.name = "test_2d_mortar_three_bodies_shared_nodes"
    solver.dump_matrices = true
    call(solver, 0.0)

    X = e3("geometry", [1.0, 1.0], 0.0)
    u = e3("displacement", [1.0, 1.0], 0.0)
    info("displacement at $X: $u")
    # code aster verification, two_elements.comm
    @test isapprox(u, [2*3.17431158889468E-02, -2.77183037855653E-01])

end
#test_2d_mortar_three_bodies_shared_nodes()


function test_auxiliary_plane_transforms()
    nodes = Vector{Float64}[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]]
    e1 = Tri3([1, 2, 3])
    # local coordinate system N, T1, T2 in node
    R = [0.0 1.0 0.0
         0.0 0.0 1.0
         1.0 0.0 0.0]
    e1["geometry"] = Vector{Float64}[nodes[1], nodes[2], nodes[3]]
    e1["normal-tangential coordinates"] = Matrix{Float64}[R, R, R]
    time::Real = 0.0
    x0, Q = create_auxiliary_plane(e1, time)
    info("x0 = $x0")
    info("Q = $Q")
    @test isapprox(x0, [1.0/3.0, 1.0/3.0, 0.0])
    @test isapprox(Q, R)
    p1 = Float64[1.0/3.0+0.1, 1.0/3.0+0.1, 1.0]
    p2 = project_point_to_auxiliary_plane(p1, x0, Q)
    info("point in auxiliary plane p2 = $p2")
    @test isapprox(p2, [0.1, 0.1])
    theta = project_point_from_plane_to_surface(p2, x0, Q, e1, time)
    info("theta = $theta")
    @test isapprox(theta[1], 0.0)
    X = e1("geometry", theta[2:3], time)
    info("projected point = $X")
    @test isapprox(X, Float64[1.0/3.0+0.1, 1.0/3.0+0.1, 0.0])
end
#test_auxiliary_plane_transforms()


function test_get_edge_intersections()
    # first case, two triangles
    S = [ 0.0 0.0; 3.0  0.0; 0.0 3.0]'
    M = [-1.0 1.0; 2.0 -0.5; 1.0 1.5]'
    P, n = get_edge_intersections(S, M)
    P_expected = [
         1.00 1.75 0.00 0.00
         0.00 0.00 0.50 1.25]
    n_expected = [
        1 1 0
        0 0 0
        1 0 1]
    @test isapprox(P, P_expected)
    @test isapprox(n, n_expected)

    # slave 4 vertices non-convex, master triangle
    S = [ 0.0 0.0; 2.5  0.0; 1.0 1.0; 0.0 2.0]'
    M = [-1.0 1.0; 2.0 -0.5; 1.0 1.5]'
    P, n = get_edge_intersections(S, M)
    P_expected = [
        1.0 1.75 1.375 0.60 0.00 0.00
        0.0 0.00 0.750 1.40 0.50 1.25]
    n_expected = [
        1 1 0
        0 1 0
        0 0 1
        1 0 1]
    @test isapprox(P, P_expected)
    @test isapprox(n, n_expected)

    # slave 3 triangle, master 4 vertices
    S = [ 0.0 0.0; 3.0  0.0; 0.0 3.0]'
    M = [-1.0 1.0; 2.0 -0.5; 1.0 1.5; -1.0 2.0]'
    P, n = get_edge_intersections(S, M)
    P_expected = [
        1.00 1.75 0.00 0.00
        0.00 0.00 0.50 1.75]
    n_expected = [
        1 1 0 0
        0 0 0 0
        1 0 1 0]
    @test isapprox(P, P_expected)
    @test isapprox(n, n_expected)
end
#test_get_edge_intersections()


function test_get_points_inside_triangle()
    S = [0.0 0.0; 3.0 0.0; 0.0 3.0]'
    pts = [-1.0 1.0; 2.0 -0.5; 1.0 1.5; 0.5 1.5]'
    P = get_points_inside_triangle(S, pts)
    @test isapprox(P, [1.0 1.5; 0.5 1.5]')
end
#test_get_points_inside_triangle()


function test_is_point_inside_convex_polygon()
    X = Vector{Float64}[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    @test is_point_inside_convex_polygon([0.5, 0.5], X) == true
    @test is_point_inside_convex_polygon([1.0, 0.5], X) == true
    @test is_point_inside_convex_polygon([1.1, 0.5], X) == false
    @test is_point_inside_convex_polygon([1.0, 1.0], X) == true
    @test is_point_inside_convex_polygon([0.0, 0.3], X) == true
    @test is_point_inside_convex_polygon([0.0, -0.000001], X) == false
end


function test_polygon_clipping_easy()
    S = [0 0; 3 0; 0 3]'
    M = [-1 1; 2 -1/2; 2 2]'
    P, n = clip_polygon(S, M)
    @test isapprox(P, [0.0 0.5; 1.0 0.0; 2.0 0.0; 2.0 1.0; 1.25 1.75; 0.0 4/3]')
    @test isapprox(n, [1 0 1; 1 1 0; 0 1 1])
end

function test_polygon_clipping_no_clip()
    # no clipping at all
    S = [-0.125   0.125  0.125  -0.125
         -0.125  -0.125  0.125   0.125]
    M = [-0.291667  -0.625     -0.625  -0.291667
         -0.208333  -0.208333   0.125   0.125   ]
    P, n = clip_polygon(S, M)
    # FIXME: check better.
    @test isa(P, Void)
    @test isa(n, Void)

end
#test_polygon_clipping_no_clip()


function test_calculate_polygon_centerpoint()
    P = [
        0.0  1.0  2.0  2.0  1.25  0.0
        0.5  0.0  0.0  1.0  1.75  1.33333]
    C = calculate_polygon_centerpoint(P)
    info("Polygon centerpoint: $C")
    @test isapprox(C, [1.0397440690338993, 0.8047003412233396])
end
#test_calculate_polygon_centerpoint()



function test_assemble_3d_problem_tri3()
    nodes = Vector{Float64}[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.1],
        [1.0, 0.0, 0.1],
        [0.0, 1.0, 0.1]]
    mel = Tri3([4, 5, 6])
    mel["geometry"] = Vector{Float64}[nodes[4], nodes[5], nodes[6]]
    sel = Tri3([1, 2, 3])
    sel["geometry"] = Vector{Float64}[nodes[1], nodes[2], nodes[3]]
#   Rv = [0.0 1.0 0.0
#         0.0 0.0 1.0
#         1.0 0.0 0.0]
#   sel["normal-tangential coordinates"] = Matrix{Float64}[Rv, Rv, Rv]
    calculate_normal_tangential_coordinates!(sel, 0.0)
    sel["master elements"] = Element[mel]
    prob = MortarProblem("temperature", 1)

    push!(prob, sel)
    stiffness_matrix = full(assemble(prob, 0.0).stiffness_matrix)
    info("stiffness matrix for this problem:\n$stiffness_matrix")
    M = D = 1/24*[2 1 1; 1 2 1; 1 1 2]
    B = [D -M]  # slave dofs are first in this.
    info("expected matrix for this problem:\n$B")
    @test isapprox(stiffness_matrix, B)

    # rotate and translate surface and check that we are still having same results
    Rx(t) = [
        1.0 0.0     0.0
        0.0 cos(t) -sin(t)
        0.0 sin(t)  cos(t)]
    Ry(t) = [
        cos(t)  0.0 sin(t)
        0.0     1.0 0.0
        -sin(t) 0.0 cos(t)
        ]
    Rz(t) = [
        cos(t) -sin(t) 0.0
        sin(t)  cos(t) 0.0
        0.0     0.0    1.0]
    T = [1.0, 1.0, 1.0]
    tx = pi/3.0
    ty = pi/4.0
    tz = pi/5.0
    for node in nodes
        node[:] = Rz(tz)*Ry(ty)*Rx(tx)*node + T
    end
    calculate_normal_tangential_coordinates!(sel, 0.0)
    stiffness_matrix = full(assemble(prob, 0.0).stiffness_matrix)
    info("sel midpnt: ", sel("geometry", [1/3, 1/3], 0.0))
    info("nt basis: ", sel("normal-tangential coordinates", [1/3, 1/3], 0.0))
    @test isapprox(stiffness_matrix, B)

end
#test_assemble_3d_problem_tri3()


function test_assemble_3d_problem_quad4()
    info("assemble 3d problem in quad4-quad4")
    nodes = Vector{Float64}[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.1],
        [2.0, 0.0, 0.1],
        [2.0, 2.0, 0.1],
        [0.0, 2.0, 0.1]]

    mel = Quad4([5, 6, 7, 8])
    mel["geometry"] = Vector{Float64}[nodes[5], nodes[6], nodes[7], nodes[8]]
    sel = Quad4([1, 2, 3, 4])
    sel["geometry"] = Vector{Float64}[nodes[1], nodes[2], nodes[3], nodes[4]]
    calculate_normal_tangential_coordinates!(sel, 0.0)
    sel["master elements"] = Element[mel]
    prob = MortarProblem("temperature", 1)

    push!(prob, sel)
    stiffness_matrix = full(assemble(prob, 0.0).stiffness_matrix)*144
    D = [16 8 4 8; 8  16 8 4;  4 8 16 8;  8 4 8 16]
    M = [25 5 1 5; 20 10 2 4; 16 8  4 8; 20 4 2 10]
    B = [D -M]  # slave dofs are first in this.
    info("expected matrix for this problem:")
    dump(round(B, 3))

    info("stiffness matrix for this problem:")
    dump(round(stiffness_matrix, 3))
    @test isapprox(stiffness_matrix, B)

end
#test_assemble_3d_problem_quad4()


function test_assemble_3d_problem_quad4_2()
    info("assemble 3d problem in quad4-quad4")
    nodes = Vector{Float64}[
        [0.0, 0.0, 0.0],
        [1/4, 0.0, 0.0],
        [1/4, 1/4, 0.0],
        [0.0, 1/4, 0.0],
        [0.0, 0.0, 0.0],
        [1/3, 0.0, 0.0],
        [1/3, 1/3, 0.0],
        [0.0, 1/3, 0.0]]

    mel = Quad4([5, 6, 7, 8])
    mel["geometry"] = Vector{Float64}[nodes[5], nodes[6], nodes[7], nodes[8]]
    sel = Quad4([1, 2, 3, 4])
    sel["geometry"] = Vector{Float64}[nodes[1], nodes[2], nodes[3], nodes[4]]
    calculate_normal_tangential_coordinates!(sel, 0.0)
    sel["master elements"] = Element[mel]
    prob = MortarProblem("temperature", 1)

    push!(prob, sel)
    stiffness_matrix = full(assemble(prob, 0.0).stiffness_matrix)*589824
    D = [
        4096 2048 1024 2048
        2048 4096 2048 1024
        1024 2048 4096 2048
        2048 1024 2048 4096
        ]
         
    M = [
        5184 1728  576 1728
        3456 3456 1152 1152
        2304 2304 2304 2304
        3456 1152 1152 3456
        ]
    B = [D -M]  # slave dofs are first in this.
    info("expected matrix for this problem:")
    dump(round(B, 3))

    info("stiffness matrix for this problem:")
    dump(round(stiffness_matrix, 3))
    @test isapprox(stiffness_matrix, B)

end
#test_assemble_3d_problem_quad4_2()


function test_assemble_3d_problem_quad4_3()
    info("assemble 3d problem in quad4-quad4")
    a = 1/4
    b = 1/3
    nodes = Vector{Float64}[
        [2*a,   a, 0],
        [3*a,   a, 0],
        [3*a, 2*a, 0],
        [2*a, 2*a, 0],
        [  b,   0, 0],
        [2*b,   0, 0],
        [2*b,   b, 0],
        [  b,   b, 0]]

    mel = Quad4([5, 6, 7, 8])
    mel["geometry"] = Vector{Float64}[nodes[5], nodes[6], nodes[7], nodes[8]]
    sel = Quad4([1, 2, 3, 4])
    sel["geometry"] = Vector{Float64}[nodes[1], nodes[2], nodes[3], nodes[4]]
    calculate_normal_tangential_coordinates!(sel, 0.0)
    sel["master elements"] = Element[mel]
    prob = MortarProblem("temperature", 1)

    push!(prob, sel)
    stiffness_matrix = full(assemble(prob, 0.0).stiffness_matrix)*186624*9
    D = [
        7904 3040 560 1456
        3040 2432 448  560
         560  448 128  160
        1456  560 160  416
        ]

    M = [
        504 1224 7956 3276
        144  720 4680  936
         18   90  990  198
         63  153 1683  693
        ]

    B = [D -M]  # slave dofs are first in this.
    info("expected matrix for this problem:")
    dump(round(B, 3))

    info("stiffness matrix for this problem:")
    dump(round(stiffness_matrix, 3))
    @test isapprox(stiffness_matrix, B)

end
#test_assemble_3d_problem_quad4_3()


function test_3d_problem()
    nodes = Vector{Float64}[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.5],
        [1.0, 0.0, 0.5],
        [1.0, 1.0, 0.5],
        [0.0, 1.0, 0.5],

        [0.0, 0.0, 0.5],
        [1.0, 0.0, 0.5],
        [1.0, 1.0, 0.5],
        [0.0, 1.0, 0.5],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ]
    el1 = Hex8([1, 2, 3, 4, 5, 6, 7, 8])
    el2 = Hex8([9, 10, 11, 12, 13, 14, 15, 16])
    sym121 = Quad4([1, 2, 3, 4])
    sym131 = Quad4([1, 2, 6, 5])
    sym132 = Quad4([9, 10, 14, 13])
    sym231 = Quad4([4, 1, 5, 8])
    sym232 = Quad4([12, 9, 13, 16])
    force = Quad4([14, 15, 16, 13])
    l2u = Quad4([5, 6, 7, 8])
    u2l = Quad4([9, 10, 11, 12])
    elements = Element[el1, el2, sym121, sym131, sym132, sym231, sym232, force, l2u, u2l]
    update!(elements, "geometry", nodes)
    el1["youngs modulus"] = el2["youngs modulus"] = 900.0
    el1["poissons ratio"] = el2["poissons ratio"] = 0.25
    sym121["displacement 3"] = 0.0
    sym131["displacement 2"] = sym132["displacement 2"] = 0.0
    sym231["displacement 1"] = sym232["displacement 1"] = 0.0
    force["displacement traction force 3"] = -100.0
    l2u["master elements"] = Element[u2l]
    calculate_normal_tangential_coordinates!(l2u, 0.0)
    fb = LinearElasticityProblem("two elastic blocks")
    push!(fb, el1, el2, force)
    bc = DirichletProblem("symmetry boundaries", "displacement", 3)
    push!(bc, sym121, sym131, sym132, sym231, sym232)
    tie = MortarProblem("tie contact between bodies", "displacement", 3)
    push!(tie, l2u)
    solver = DirectSolver("solution of elasticity problem")
    push!(solver, fb)
    push!(solver, bc)
    push!(solver, tie)
    solver.nonlinear_problem = false
    solver.method = :UMFPACK
    call(solver, 0.0)
    X = el2("geometry", [1.0, 1.0, 1.0], 0.0)
    u = el2("displacement", [1.0, 1.0, 1.0], 0.0)
    info("displacement at $X = $u")
    @test isapprox(u, 1/36*[1, 1, -4])
end
#test_3d_problem()

#=
@testset "plane quad4 projector tests" begin
    a = 1/2
    b = 1/3
    nodes = Dict{Int64, Vector{Float64}}(
        1 => [0.0, 0.0, 0.0],
        2 => [1/2, 0.0, 0.0],
        3 => [1.0, 0.0, 0.0],
        4 => [0.0, 1.0, 0.0],
        5 => [1/2, 1.0, 0.0],
        6 => [1.0, 1.0, 0.0],

        7 => [0.0, 0.0, 0.0],
        8 => [1/3, 0.0, 0.0],
        9 => [2/3, 0.0, 0.0],
       10 => [1.0, 0.0, 0.0],
       11 => [0.0, 1/2, 0.0],
       12 => [1/3, 1/2, 0.0],
       13 => [2/3, 1/2, 0.0],
       14 => [1.0, 1/2, 0.0],
       15 => [0.0, 1.0, 0.0],
       16 => [1/3, 1.0, 0.0],
       17 => [2/3, 1.0, 0.0],
       18 => [1.0, 1.0, 0.0],
    )
    sel1 = Quad4([1, 2, 5, 4])
    sel2 = Quad4([2, 3, 6, 5])
    mel1 = Quad4([7, 8, 12, 11])
    mel2 = Quad4([8, 9, 13, 12])
    mel3 = Quad4([9, 10, 14, 13])
    mel4 = Quad4([11, 12, 16, 15])
    mel5 = Quad4([12, 13, 17, 16])
    mel6 = Quad4([13, 14, 18, 17])
    update(Element[sel1, sel2, mel1, mel2, mel3, mel4, mel5, mel6], "geometry", nodes)
    calculate_normal_tangential_coordinates!(sel1, 0.0)
    calculate_normal_tangential_coordinates!(sel2, 0.0)
    prob = MortarProblem("temperature", 1)
    push!(prob, sel1)
    push!(prob, sel2)

    sel1["master elements"] = [mel1, mel2, mel4, mel5]
    sel2["master elements"] = [mel2, mel3, mel5, mel6]
    stiffness_matrix = full(assemble(prob, 0.0).stiffness_matrix)*2592*6
    info("interface matrix:")
    dump(round(stiffness_matrix, 3))
    B = [
        864  432   0 432  216   0 -420 -375  -15    0 -504 -450  -18    0  -84  -75   -3    0
        432 1728 432 216  864 216 -120 -690 -690 -120 -144 -828 -828 -144  -24 -138 -138  -24
          0  432 864   0  216 432    0  -15 -375 -420    0  -18 -450 -504    0   -3  -75  -84
        432  216   0 864  432   0  -84  -75   -3    0 -504 -450  -18    0 -420 -375  -15    0
        216  864 216 432 1728 432  -24 -138 -138  -24 -144 -828 -828 -144 -120 -690 -690 -120
          0  216 432   0  432 864    0   -3  -75  -84    0  -18 -450 -504    0  -15 -375 -420
        ]
    info("expected interface matrix:")
    dump(round(B, 3))
    @test isapprox(stiffness_matrix, B)
end
=#

@testset "plane quad4 projector master 3x3 slave 2x2" begin
    a = 1/2
    b = 1/3
    nodes = Dict{Int64, Vector{Float64}}(
        1 => [0*a, 0*a, 0.0],
        2 => [1*a, 0*a, 0.0],
        3 => [2*a, 0*a, 0.0],
        4 => [0*a, 1*a, 0.0],
        5 => [1*a, 1*a, 0.0],
        6 => [2*a, 1*a, 0.0],
        7 => [0*a, 2*a, 0.0],
        8 => [1*a, 2*a, 0.0],
        9 => [2*a, 2*a, 0.0],

       10 => [0*b, 0*b, 0.0],
       11 => [1*b, 0*b, 0.0],
       12 => [2*b, 0*b, 0.0],
       13 => [3*b, 0*b, 0.0],
       14 => [0*b, 1*b, 0.0],
       15 => [1*b, 1*b, 0.0],
       16 => [2*b, 1*b, 0.0],
       17 => [3*b, 1*b, 0.0],
       18 => [0*b, 2*b, 0.0],
       19 => [1*b, 2*b, 0.0],
       20 => [2*b, 2*b, 0.0],
       21 => [3*b, 2*b, 0.0],
       22 => [0*b, 3*b, 0.0],
       23 => [1*b, 3*b, 0.0],
       24 => [2*b, 3*b, 0.0],
       25 => [3*b, 3*b, 0.0],
    )
    sel1 = Quad4([1, 2, 5, 4])
    sel2 = Quad4([2, 3, 6, 5])
    sel3 = Quad4([4, 5, 8, 7])
    sel4 = Quad4([5, 6, 9, 8])
    mel1 = Quad4([10, 11, 15, 14])
    mel2 = Quad4([11, 12, 16, 15])
    mel3 = Quad4([12, 13, 17, 16])
    mel4 = Quad4([14, 15, 19, 18])
    mel5 = Quad4([15, 16, 20, 19])
    mel6 = Quad4([16, 17, 21, 20])
    mel7 = Quad4([18, 19, 23, 22])
    mel8 = Quad4([19, 20, 24, 23])
    mel9 = Quad4([20, 21, 25, 24])
    update!(Element[sel1, sel2, sel3, sel4, mel1, mel2, mel3,
                    mel4, mel5, mel6, mel7, mel8, mel9], "geometry", nodes)
    calculate_normal_tangential_coordinates!(sel1, 0.0)
    calculate_normal_tangential_coordinates!(sel2, 0.0)
    calculate_normal_tangential_coordinates!(sel3, 0.0)
    calculate_normal_tangential_coordinates!(sel4, 0.0)
    prob = MortarProblem("temperature", 1)
    push!(prob, sel1)
    push!(prob, sel2)
    push!(prob, sel3)
    push!(prob, sel4)

    master_elements = [mel1, mel2, mel3, mel4, mel5, mel6, mel7, mel8, mel9]
    sel1["master elements"] = master_elements
    sel2["master elements"] = master_elements
    sel3["master elements"] = master_elements
    sel4["master elements"] = master_elements
    B = sparse(assemble(prob, 0.0).stiffness_matrix, 25, 25)*46656
    B = full(B)
    D = B[1:9,1:9]
    M = B[1:9,10:end]
    info("interface matrix D:")
    dump(round(D, 3))
    info("interface matrix M:")
    dump(round(M, 3))
    D_expected = [
        1296  648    0  648  324    0    0    0    0
         648 2592  648  324 1296  324    0    0    0
           0  648 1296    0  324  648    0    0    0
         648  324    0 2592 1296    0  648  324    0
         324 1296  324 1296 5184 1296  324 1296  324
           0  324  648    0 1296 2592    0  324  648
           0    0    0  648  324    0 1296  648    0
           0    0    0  324 1296  324  648 2592  648
           0    0    0    0  324  648    0  648 1296]

    M_expected = [
        -784  -700   -28    0  -700  -625   -25     0   -28   -25    -1     0    0     0     0    0
        -224 -1288 -1288 -224  -200 -1150 -1150  -200    -8   -46   -46    -8    0     0     0    0
           0   -28  -700 -784     0   -25  -625  -700     0    -1   -25   -28    0     0     0    0
        -224  -200    -8    0 -1288 -1150   -46     0 -1288 -1150   -46     0 -224  -200    -8    0
         -64  -368  -368  -64  -368 -2116 -2116  -368  -368 -2116 -2116  -368  -64  -368  -368  -64
           0    -8  -200 -224     0   -46 -1150 -1288     0   -46 -1150 -1288    0    -8  -200 -224
           0     0     0    0   -28   -25    -1     0  -700  -625   -25     0 -784  -700   -28    0
           0     0     0    0    -8   -46   -46    -8  -200 -1150 -1150  -200 -224 -1288 -1288 -224
           0     0     0    0     0    -1   -25   -28     0   -25  -625  -700    0   -28  -700 -784]


    info("D - D_expected")
    dump(D - D_expected)
    info("M - M_expected")
    dump(M - M_expected)
    
    @test isapprox(D, D_expected)
    @test isapprox(M, M_expected)
    #=
    B = [
        864  432   0 432  216   0 -420 -375  -15    0 -504 -450  -18    0  -84  -75   -3    0
        432 1728 432 216  864 216 -120 -690 -690 -120 -144 -828 -828 -144  -24 -138 -138  -24
          0  432 864   0  216 432    0  -15 -375 -420    0  -18 -450 -504    0   -3  -75  -84
        432  216   0 864  432   0  -84  -75   -3    0 -504 -450  -18    0 -420 -375  -15    0
        216  864 216 432 1728 432  -24 -138 -138  -24 -144 -828 -828 -144 -120 -690 -690 -120
          0  216 432   0  432 864    0   -3  -75  -84    0  -18 -450 -504    0  -15 -375 -420
        ]
    info("expected interface matrix:")
    dump(round(B, 3))
    @test isapprox(stiffness_matrix, B)
=#
end

end
