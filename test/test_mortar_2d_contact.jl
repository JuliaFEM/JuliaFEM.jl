# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM.Test

using JuliaFEM.Core: Node, Seg2, Tri3, update!, calculate_normal_tangential_coordinates!,
                     PlaneStressLinearElasticityProblem, DirichletProblem, MortarProblem,
                     get_elements, DirectSolver, calculate_nodal_vector, FieldAssembly,
                     FieldProblem, set_linear_system_solver!, set_nonlinear_max_iterations!

import JuliaFEM.Core: postprocess_assembly!, linear_system_solver_preprocess!

function postprocess_assembly!(assembly::FieldAssembly, problem::FieldProblem{PlaneStressLinearElasticityProblem}, time::Real)
    f = full(assembly.force_vector)
    info("force vector = $(f')")
end

function linear_system_solver_preprocess!(solver, iter, time, K, f, C1, C2, D, g, sol, la, ::Type{Val{:foo}})
    info("stiffness matrix")
    dump(round(full(K), 3))
    info("constraint matrix C1")
    dump(round(full(C1), 3))
    info("constraint matrix C2")
    dump(round(full(C2), 3))
    info("force vector")
    dump(round(full(f)', 3))
    info("constraint vector")
    dump(round(full(g)', 3))

end

macro debug(msg)
    haskey(ENV, "DEBUG") || return
    return msg
end

@testset "2d frictionless contact" begin
    nodes = Node[
        [6.0, 6.0],
        [7.0, 8.0],
        [0.0, 0.0],
        [6.0, 0.0],
        [7.0, 0.0],
        [19.0, 0.0]]
    fel1 = Tri3([1, 3, 4])
    fel2 = Tri3([2, 5, 6])
    force = Seg2([3, 1])
    bnd1 = Seg2([3, 4])
    bnd2 = Seg2([5, 6])
    sel = Seg2([1, 4])
    mel = Seg2([5, 2])
    update!([fel1, fel2, force, sel, mel, bnd1, bnd2], "geometry", nodes)

    prob = PlaneStressLinearElasticityProblem()
    push!(prob, fel1, fel2)
    push!(prob, force)
    update!([fel1, fel2], "youngs modulus", 90.0)
    update!([fel1, fel2], "poissons ratio", 0.25)
    update!([force], "displacement traction force 1", 6/sqrt(2))
    fel1["displacement nodal load"] = Vector{Float64}[[0.0, 0.0], [0.0, 0.0], [18.0, 0.0]]

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
#   push!(solver, cont)
    set_linear_system_solver!(solver, :UMFPACK)
    set_nonlinear_max_iterations!(solver, 2)
    push!(solver.linear_system_solver_preprocessors, :foo)
    time = 0.0
    call(solver, time)

    @debug begin
        u = calculate_nodal_vector("displacement", 2, get_elements(prob), time)
        info("solution vector")
        dump(reshape(round(u, 8), 2, 6))
        la = calculate_nodal_vector("reaction force", 2, get_elements(prob), time)
        info("reaction force")
        dump(reshape(round(la, 8), 2, 6))
    end
end

