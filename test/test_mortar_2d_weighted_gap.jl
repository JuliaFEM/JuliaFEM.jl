# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using Test

#= TODO: Fix test.
@testset "calculate mortar matrices and weighted gap vector for 2d model" begin
    nodes = Node[
        [1.0, 1.0],
        [3.0, 1.0],
        [2.0, 2.0],
        [4.0, 4.0]]
    sel = Seg2([1, 2])
    mel = Seg2([3, 4])
    update!([sel, mel], "geometry", nodes)
    calculate_normal_tangential_coordinates!(sel, 0.0)
    sel["master elements"] = [mel]
    prob = MortarProblem("displacement", 2)
    push!(prob, sel)
    a = assemble(prob, 0.0)
    C1 = full(a.C1)
    D = C1[:, [1, 2, 3, 4]]*24
    M = C1[:, [5, 6, 7, 8]]*24
    X = calculate_nodal_vector("geometry", 2, [sel, mel], 0.0)
    @debug begin
        @info("nodal vector")
        dump(round(full(X), 3))
    end
    g = -C1*X
    g = full(g)
    @debug begin
        dump(round(D, 3))
        dump(round(M, 3))
        dump(round(g, 3))
    end
    I = eye(2)
    @test isapprox(D, [2*I 4*I; 4*I 14*I])
    @test isapprox(-M, [5*I 1*I; 13*I 5*I])
    @test isapprox(g, 1/6*[0, 2, 0, 7])
end
=#
