# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Testing
using JuliaFEM: calculate_normals

function get_test_2d_model()
    X = Dict(
             7 => [0.0, 1.0],
             8 => [5/4, 1.0],
             9 => [2.0, 1.0],
             10 => [0.0, 1.0],
             11 => [3/4, 1.0],
             12 => [2.0, 1.0])
    mel1 = Element(Seg2, [7, 8])
    mel2 = Element(Seg2, [8, 9])
    sel1 = Element(Seg2, [10, 11])
    sel2 = Element(Seg2, [11, 12])
    update!([mel1, mel2, sel1, sel2], "geometry", X)
    update!([sel1, sel2], "master elements", [sel1, sel2])
    slave_elements = [sel1, sel2]
    time = 0.0
    normals, tangents = calculate_normals(slave_elements, time, Val{1})
    update!(slave_elements, "normal", time => normals)
    return [sel1, sel2], [mel1, mel2]
end

@testset "calculate flat 2d projection from slave to master" begin
    (sel1, sel2), (mel1, mel2) = get_test_2d_model()

    time = 0.0
    X1 = sel1("geometry", [-1.0], time)
    n1 = sel1("normal", [-1.0], time)
    println("X1 = ", X1)
    println("n1 = ", n1)
    xi2 = project_from_slave_to_master(mel1, X1, n1, time)
    @test isapprox(xi2, -1.0)

    X1 = sel1("geometry", [1.0], time)
    n1 = sel1("normal", [1.0], time)
    xi2 = project_from_slave_to_master(mel1, X1, n1, time)
    @test isapprox(xi2, 0.2)

    X2 = mel1("geometry", xi2, time)
    @test isapprox(X2, [3/4, 1.0])
end

@testset "calculate flat 2d projection from master to slave" begin
    (sel1, sel2), (mel1, mel2) = get_test_2d_model()
    time = 0.0
    x2 = mel1("geometry", [-1.0], time)
    xi1 = project_from_master_to_slave(sel1, x2, time)
    @test isapprox(xi1, -1.0)
    x2 = mel1("geometry", [1.0], time)
    xi1 = project_from_master_to_slave(sel1, x2, time)
    X1 = sel1("geometry", xi1, time)
    @test isapprox(X1, [5/4, 1.0])
end

@testset "calculate flat 2d projection rotated 90 degrees" begin
    X = Dict{Int64, Vector{Float64}}(
        1 => [0.0, 0.0],
        2 => [0.0, 1.0],
        3 => [0.0, 1.0],
        4 => [0.0, 0.0])
    sel1 = Element(Seg2, [1, 2])
    mel1 = Element(Seg2, [3, 4])
    update!([sel1, mel1], "geometry", X)
    time = 0.0
    slave_elements = [sel1]
    time = 0.0
    normals, tangents = calculate_normals(slave_elements, time, Val{1})
    update!(slave_elements, "normal", time => normals)

    X2 = mel1("geometry", [-1.0], time)
    xi = project_from_master_to_slave(sel1, X2, time)
    @test isapprox(xi, 1.0)

    X2 = mel1("geometry", [1.0], time)
    xi = project_from_master_to_slave(sel1, X2, time)
    @test isapprox(xi, -1.0)

    X1 = sel1("geometry", [-1.0], time)
    n1 = sel1("normal", [-1.0], time)
    xi = project_from_slave_to_master(mel1, X1, n1, time)
    @test isapprox(xi, 1.0)

    X1 = sel1("geometry", [1.0], time)
    n1 = sel1("normal", [1.0], time)
    xi = project_from_slave_to_master(mel1, X1, n1, time)
    @test isapprox(xi, -1.0)
end

