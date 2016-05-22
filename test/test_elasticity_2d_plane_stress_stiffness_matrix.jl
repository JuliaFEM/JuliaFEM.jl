# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# http://ahojukka5.github.io/posts/finite-element-solution-for-one-element-problem/

using JuliaFEM
using JuliaFEM.Test

@testset "test 2d linear elasticity local matrices" begin
    element = Element(Quad4)
    element["geometry"] = Vector{Float64}[
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]]
    element["youngs modulus"] = 288.0
    element["poissons ratio"] = 1/3
    element["displacement load"] = DCTI([4.0, 8.0])

    problem = Problem(Elasticity, "[0x1] x [0x1] block", 2)
    problem.properties.formulation = :plane_stress
    K, f = assemble(problem, element)

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
