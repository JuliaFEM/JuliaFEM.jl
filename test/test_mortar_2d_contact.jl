# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM.Test

using JuliaFEM.Core: Node, Seg2, Tri3, update!, calculate_normal_tangential_coordinates!,
                     PlaneStressLinearElasticityProblem, DirichletProblem, MortarProblem,
                     get_elements, DirectSolver

macro debug(msg)
    haskey(ENV, "DEBUG") || return
    return msg
end

@testset "2d frictionless contact" begin
    nodes = Node[
        [3.0, 3.0],
        [4.0, 6.0],
        [0.0, 0.0],
        [3.0, 0.0],
        [4.0, 0.0],
        [10.0, 0.0]]
    fel1 = Tri3([3, 4, 1])
    fel2 = Tri3([5, 6, 2])
    bnd1 = Seg2([3, 4])
    bnd2 = Seg2([5, 6])
    sel = Seg2([1, 4])
    mel = Seg2([5, 2])
    update!([fel1, fel2, sel, mel, bnd1, bnd2], "geometry", nodes)

    prob = PlaneStressLinearElasticityProblem()
    push!(prob, fel1, fel2)
    update!(get_elements(prob), "youngs modulus", 90.0)
    update!(get_elements(prob), "poissons ratio", 0.25)

    bc = DirichletProblem("displacement", 2)
    push!(bc, bnd1, bnd2)
    update!(get_elements(bc), "displacement", 0.0 => Vector{Float64}[[0.0, 0.0], [0.0, 0.0]])

    cont = MortarProblem("displacement", 2)
    push!(cont, sel, mel)
    calculate_normal_tangential_coordinates!(sel, 0.0)
    sel["master elements"] = [mel]

    @debug begin
        info("fel1.fields = $(fel1.fields)")
        info("bnd1.fields = $(bnd1.fields)")
    end

    solver = DirectSolver()
    push!(solver, prob)
    push!(solver, bc)
    push!(solver, cont)
    solver.method = :UMFPACK
    solver.max_iterations = 5
    call(solver, 0.0)
end

