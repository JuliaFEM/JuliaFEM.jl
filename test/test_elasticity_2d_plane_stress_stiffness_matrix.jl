# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# http://ahojukka5.github.io/posts/finite-element-solution-for-one-element-problem/

using JuliaFEM
using JuliaFEM.Test

@testset "test 2d linear elasticity local matrices" begin
    element = Element(Quad4, [1, 2, 3, 4])
    X = Dict{Int64, Vector{Float64}}(
        1 => [0.0, 0.0],
        2 => [1.0, 0.0],
        3 => [1.0, 1.0],
        4 => [0.0, 1.0])
    update!(element, "geometry", X)
    update!(element, "youngs modulus" =>  288.0, "poissons ratio" => 1/3)
    element["displacement load"] = DCTI([4.0, 8.0])

    problem = Problem(Elasticity, "[0x1] x [0x1] block", 2)
    problem.properties.formulation = :plane_stress
    assemble!(problem, element)
    K = full(problem.assembly.K)
    f = full(problem.assembly.f)

    K_expected = [
        144  54 -90   0 -72 -54  18   0
         54 144   0  18 -54 -72   0 -90
        -90   0 144 -54  18   0 -72  54
          0  18 -54 144   0 -90  54 -72
        -72 -54  18   0 144  54 -90   0
        -54 -72   0 -90  54 144   0  18
         18   0 -72  54 -90   0 144 -54
          0 -90  54 -72   0  18 -54 144]

    f_expected = [1, 2, 1, 2, 1, 2, 1, 2]

    @test isapprox(K, K_expected)
    @test isapprox(f, f_expected)
end
