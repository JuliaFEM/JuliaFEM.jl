# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using SparseArrays, Test

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
    @info("x0 = $x0")
    @info("Q = $Q")
    @test isapprox(x0, [1.0/3.0, 1.0/3.0, 0.0])
    @test isapprox(Q, R)
    p1 = Float64[1.0/3.0+0.1, 1.0/3.0+0.1, 1.0]
    p2 = project_point_to_auxiliary_plane(p1, x0, Q)
    @info("point in auxiliary plane p2 = $p2")
    @test isapprox(p2, [0.1, 0.1])
    theta = project_point_from_plane_to_surface(p2, x0, Q, e1, time)
    @info("theta = $theta")
    @test isapprox(theta[1], 0.0)
    X = e1("geometry", theta[2:3], time)
    @info("projected point = $X")
    @test isapprox(X, Float64[1.0/3.0+0.1, 1.0/3.0+0.1, 0.0])
end

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


function test_get_points_inside_triangle()
    S = [0.0 0.0; 3.0 0.0; 0.0 3.0]'
    pts = [-1.0 1.0; 2.0 -0.5; 1.0 1.5; 0.5 1.5]'
    P = get_points_inside_triangle(S, pts)
    @test isapprox(P, [1.0 1.5; 0.5 1.5]')
end


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


function test_calculate_polygon_centerpoint()
    P = [
        0.0  1.0  2.0  2.0  1.25  0.0
        0.5  0.0  0.0  1.0  1.75  1.33333]
    C = calculate_polygon_centerpoint(P)
    @info("Polygon centerpoint: $C")
    @test isapprox(C, [1.0397440690338993, 0.8047003412233396])
end



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
    @info("stiffness matrix for this problem:\n$stiffness_matrix")
    M = D = 1/24*[2 1 1; 1 2 1; 1 1 2]
    B = [D -M]  # slave dofs are first in this.
    @info("expected matrix for this problem:\n$B")
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
    @info("sel midpnt: ", sel("geometry", [1/3, 1/3], 0.0))
    @info("nt basis: ", sel("normal-tangential coordinates", [1/3, 1/3], 0.0))
    @test isapprox(stiffness_matrix, B)

end


function test_assemble_3d_problem_quad4()
    @info("assemble 3d problem in quad4-quad4")
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
    @info("expected matrix for this problem:")
    dump(round(B, 3))

    @info("stiffness matrix for this problem:")
    dump(round(stiffness_matrix, 3))
    @test isapprox(stiffness_matrix, B)

end


function test_assemble_3d_problem_quad4_2()
    @info("assemble 3d problem in quad4-quad4")
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
    @info("expected matrix for this problem:")
    dump(round(B, 3))

    @info("stiffness matrix for this problem:")
    dump(round(stiffness_matrix, 3))
    @test isapprox(stiffness_matrix, B)

end


function test_assemble_3d_problem_quad4_3()
    @info("assemble 3d problem in quad4-quad4")
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
    @info("expected matrix for this problem:")
    dump(round(B, 3))

    @info("stiffness matrix for this problem:")
    dump(round(stiffness_matrix, 3))
    @test isapprox(stiffness_matrix, B)

end


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
    solver()
    X = el2("geometry", [1.0, 1.0, 1.0], 0.0)
    u = el2("displacement", [1.0, 1.0, 1.0], 0.0)
    @info("displacement at $X = $u")
    @test isapprox(u, 1/36*[1, 1, -4])
end

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
    @info("interface matrix:")
    dump(round(stiffness_matrix, 3))
    B = [
        864  432   0 432  216   0 -420 -375  -15    0 -504 -450  -18    0  -84  -75   -3    0
        432 1728 432 216  864 216 -120 -690 -690 -120 -144 -828 -828 -144  -24 -138 -138  -24
          0  432 864   0  216 432    0  -15 -375 -420    0  -18 -450 -504    0   -3  -75  -84
        432  216   0 864  432   0  -84  -75   -3    0 -504 -450  -18    0 -420 -375  -15    0
        216  864 216 432 1728 432  -24 -138 -138  -24 -144 -828 -828 -144 -120 -690 -690 -120
          0  216 432   0  432 864    0   -3  -75  -84    0  -18 -450 -504    0  -15 -375 -420
        ]
    @info("expected interface matrix:")
    dump(round(B, 3))
    @test isapprox(stiffness_matrix, B)
end
=#

#= TODO: Fix test.
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
    @info("interface matrix D:")
    dump(round(D, 3))
    @info("interface matrix M:")
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


    @info("D - D_expected")
    dump(D - D_expected)
    @info("M - M_expected")
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
    @info("expected interface matrix:")
    dump(round(B, 3))
    @test isapprox(stiffness_matrix, B)
=#
end
=#
