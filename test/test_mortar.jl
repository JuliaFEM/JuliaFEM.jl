# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

module MortarTests

using JuliaFEM
using JuliaFEM.Test

function test_calc_flat_2d_assembly()
    # this is hand calculated and given example in my thesis
    N = Vector[
        [0.0, 2.0], [1.0, 2.0], [2.0, 2.0],
        [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
        [0.0, 1.0], [5/4, 1.0], [2.0, 1.0],
        [0.0, 1.0], [3/4, 1.0], [2.0, 1.0]]

    slave1 = Seg2([10, 11])
    slave1["geometry"] = Vector[N10, N11]

    slave2 = Seg2([11, 12])
    slave2["geometry"] = Vector[N11, N12]

    master1 = Seg2([7, 8])
    master1["geometry"] = Vector[N7, N8]

    master2 = Seg2([8, 9])
    master2["geometry"] = Vector[N8, N9]

    problem = MortarProblem()
    push!(problem, slave1)
    push!(problem, slave2)
    push!(problem.master_elements, master1)
    push!(problem.master_elements, master2)

    rotation_matrix(phi) = [cos(phi) -sin(phi); sin(phi) cos(phi)]
    # should be n = [0 -1]' and t = [1 0]'
    @test isapprox(rotation_matrix(-phi/2), [[0 -1]' [1 0]'])

    problem.node_csys = Dict(
        10 => rotation_matrix(-phi/2),
        11 => rotation_matrix(-phi/2),
        12 => rotation_matrix(-phi/2))

    # first index = master element id
    # second index = slave element id
    problem.element_pairs = zeros(2, 2)
    # first slave element connects to master element 1
    problem.element_pairs[1, 1] = true
    # second slave element connects to master element 1
    problem.element_pairs[1, 2] = true
    # second slave element connects to master element 2
    problem.element_pairs[2, 2] = true

    B_expected = zeros(12, 9)

    S1 = [10, 11]
    M1 = [7, 8]
    B_expected[S1,S1] += [1/4 1/8; 1/8 1/4]
    B_expected[S1,M1] += [3/10 3/40; 9/40 3/20]

    la = initialize_local_assembly(problem)
    calculate_local_assembly!(la, problem.equations[1], "reaction force", 0.0, problem=problem)
    B = full(la.lhs)
    @test isapprox(B, B_expected) 

    fill!(B_expected, 0.0)

    S2 = [11, 12]
    M2 = [7, 8]
    B_expected[S2,S2] += [49/150 11/150; 11/150 2/75]
    B_expected[S2,M2] += [13/150 47/150; 1/75 13/150]
    S3 = [11, 12]
    M3 = [8, 9]
    B_expected[S3,S3] += [9/100 27/200; 27/200 39/100]
    B_expected[S3,M3] += [3/20 3/40; 9/40 3/10]

    la = initialize_local_assembly(problem)
    calculate_local_assembly!(la, problem.equations[1], "reaction force", 0.0, problem=problem)
    B = full(la.lhs)
    @test isapprox(B, B_expected)
end

end
