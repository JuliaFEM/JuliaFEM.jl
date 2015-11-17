# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

module MortarTests

using JuliaFEM
using JuliaFEM.Test

using JuliaFEM: MSeg2, Seg2, MortarProblem, MortarEquation, MortarElement, Assembly, assemble!
using JuliaFEM: get_basis, grad, project_from_slave_to_master, project_from_master_to_slave

function get_test_2d_model()
    # this is hand calculated and given as an example in my thesis
    N = Vector[
        [0.0, 2.0], [1.0, 2.0], [2.0, 2.0],
        [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
        [0.0, 1.0], [5/4, 1.0], [2.0, 1.0],
        [0.0, 1.0], [3/4, 1.0], [2.0, 1.0]]
    rotation_matrix(phi) = [cos(phi) -sin(phi); sin(phi) cos(phi)]
    slave1 = MSeg2([10, 11])
    slave1["geometry"] = Vector[N[10], N[11]]
    # should be n = [0 -1]' and t = [1 0]'
    slave1["nodal ntsys"] = Matrix[rotation_matrix(-pi/2), rotation_matrix(-pi/2)]
    slave2 = MSeg2([11, 12])
    slave2["geometry"] = Vector[N[11], N[12]]
    # should be n = [0 -1]' and t = [1 0]'
    slave2["nodal ntsys"] = Matrix[rotation_matrix(-pi/2), rotation_matrix(-pi/2)]
    master1 = MSeg2([7, 8])
    master1["geometry"] = Vector[N[7], N[8]]
    master2 = MSeg2([8, 9])
    master2["geometry"] = Vector[N[8], N[9]]
    push!(slave1.master_elements, master1)
    push!(slave1.master_elements, master2)
    push!(slave2.master_elements, master1)
    push!(slave2.master_elements, master2)
    return [slave1, slave2], [master1, master2]
end


function test_calc_flat_2d_projection()
    slaves, masters = get_test_2d_model()
    slave1, slave2 = slaves
    master1, master2 = masters

    xi2a = project_from_slave_to_master(slave1, master1, [-1.0])
    @test xi2a == [-1.0]

    xi2b = project_from_slave_to_master(slave1, master1, [1.0])
    @test xi2b == [ 0.2]
    X2 = get_basis(master1)("geometry", xi2b)
    @test X2 == [3/4, 1.0]

    xi1a = project_from_master_to_slave(slave1, master1, [-1.0])
    @test xi1a == [-1.0]
    xi1b = project_from_master_to_slave(slave1, master1, [1.0])
    X1 = get_basis(slave1)("geometry", xi1b)
    @test X1 == [5/4, 1.0]
end

function test_create_flat_2d_assembly()
    slaves, masters = get_test_2d_model()
    slave1, slave2 = slaves
    master1, master2 = masters

    info("creating problem")
    problem = MortarProblem()
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
    assemble!(assembly, problem.equations[1], 0.0, problem)
    B = round(full(assembly.lhs, 12, 12), 6)
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
    assemble!(assembly, problem.equations[2], 0.0, problem)
    B = full(assembly.lhs)
    info("size of B = $(size(B))")
    info("B matrix in second slave element = \n$(B[11:12,:])")
    info("B matrix expected = \n$(B_expected[11:12,:])")

    @test isapprox(B, B_expected)
end

function test_patch_test_heat_2d()


    
    slaves, masters = get_test_2d_model()
    problem2 = MortarProblem()
    for slave in slaves:
        push!(problem2, slave)
    end

end

end
