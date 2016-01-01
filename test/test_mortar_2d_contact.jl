# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM.Test

using JuliaFEM.Core: Node, Seg2, Tri3, update!, calculate_normal_tangential_coordinates!,
                     PlaneStressLinearElasticityProblem, DirichletProblem, MortarProblem,
                     get_elements, DirectSolver, calculate_nodal_vector, FieldAssembly,
                     FieldProblem, set_linear_system_solver!, set_nonlinear_max_iterations!,
                     get_elements, Element, BoundaryAssembly, BoundaryProblem,
                     get_integration_points, get_jacobian, get_connectivity, StandardBasis,
                     add_postprocessor!, add_preprocessor!

import JuliaFEM.Core: assemble_postprocess!, linear_system_solver_preprocess!

function calculate_normal_tangential_coordinates(elements::Vector{Element}, time::Real)
    P = SparseMatrixCOO()
    field_dim = 2
    for element in elements
        haskey(element, "normal-tangential coordinates") || continue
        for ip in get_integration_points(element, Val{2})
            J = get_jacobian(element, ip, time)
            w = ip.weight*norm(J)
            nt = transpose(element("normal-tangential coordinates", ip, time))
            normal = nt[1,:]
            tangent = nt[2,:]
            for nid in get_connectivity(element)
                ndofs = [2*(nid-1)+1, 2*(nid-1)+2]
                add!(P, [2*(nid-1)+1], ndofs, normal)
                add!(P, [2*(nid-1)+2], ndofs, tangent)
            end
        end
    end
    P = sparse(P)
    for i=1:size(P,1)
        n = norm(P[i,:])
        if n > 0.0
            P[i,:] = P[i,:] / n
        end
    end
    return P
end

function assemble_postprocess!(assembly, problem, time::Real, ::Type{Val{:remove_tangential_constraints}})
    # example how to use postprocessor to manipulate constraint matrix before summing assemblies together
    info("postprocess mortar assembly: remove contraints in tangent direction on boundary.")
    C1 = sparse(assembly.C1)
    C2 = sparse(assembly.C2)
    dim = size(C1, 1)
    P = calculate_normal_tangential_coordinates(get_elements(problem), time)
    info("projection matrix for normals: ")
    dump(round(full(P), 3))
    C1 = P*C1
    C2 = P*C2
    for i=2:2:dim
        C1[i,:] = 0
        C2[i,:] = 0
    end
    assembly.C1 = C1
    assembly.C2 = C2
    info("postprocess mortar assembly: done.")
end

function assemble_postprocess!(assembly, problem, time::Real, ::Type{Val{:remove_constraint_from_dofs}}, dofs)
    # giving additional arguments and keywords is possible too
    C1 = sparse(assembly.C1)
    C2 = sparse(assembly.C2)
    info("removing constraints from dofs: $dofs")
    info(round(full(C1), 3))
    for d in dofs
        C1[d,:] = 0
        C2[d,:] = 0
    end
    assembly.C1 = C1
    assembly.C2 = C2
end

function linear_system_solver_preprocess!(solver, iter, time, K, f, C1, C2, D, g, sol, la, ::Type{Val{:foo}})
    # example how to use preprocessor to dump matrices before solution
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
    gap = [0.0, 0.0]
    nodes = Node[
        [6.0, 6.0],
        [6.0, 8.0]+gap,
        [0.0, 0.0],
        [6.0, 0.0],
        [6.0, 0.0]+gap,
        [18.0, 0.0]+gap]
    fel1 = Tri3([1, 3, 4])
    fel2 = Tri3([2, 5, 6])
    force = Seg2([3, 1])
    bnd1 = Seg2([3, 4])
    bnd2 = Seg2([5, 6])
    sel = Seg2([1, 4])
    mel = Seg2([5, 2])
    update!([fel1, fel2, force, sel, mel], "geometry", nodes)
    update!([bnd1, bnd2], "geometry", nodes)

    prob = FieldProblem(PlaneStressLinearElasticityProblem, "bodies", 2)
    push!(prob, fel1, fel2)
    push!(prob, force)
    update!([fel1, fel2], "youngs modulus", 90.0)
    update!([fel1, fel2], "poissons ratio", 0.25)
    update!([force], "displacement traction force 1", 6/sqrt(2))

    bc = BoundaryProblem(DirichletProblem, "support", "displacement", 2)
    push!(bc, bnd1, bnd2)
    update!(get_elements(bc), "displacement", 0.0 => Vector{Float64}[[0.0, 0.0], [0.0, 0.0]])

    cont = BoundaryProblem(MortarProblem, "contact", "displacement", 2)
    push!(cont, sel, mel)
    calculate_normal_tangential_coordinates!(sel, 0.0)
    nt = sel("normal-tangential coordinates", [0.0], 0.0)
    info("normal direction = $(nt)")
    sel["master elements"] = [mel]
    # remove coefficients from node 4 (dofs 7-8) because this conflicts with dirichlet bc.
    add_postprocessor!(cont, :remove_constraint_from_dofs, [7, 8])

    @debug begin
        info("fel1.fields = $(fel1.fields)")
    end

    solver = DirectSolver()
    push!(solver, prob)
    push!(solver, bc)
    push!(solver, cont)
    set_linear_system_solver!(solver, :UMFPACK)
    set_nonlinear_max_iterations!(solver, 2)
    push!(solver.linear_system_solver_preprocessors, :foo)
    time = 0.0
    call(solver, time)

    @debug begin
        u = calculate_nodal_vector("displacement", 2, get_elements(prob), time)
        info("solution vector")
        dump(reshape(round(u, 8), 2, 6))
#       la = calculate_nodal_vector("reaction force", 2, get_elements(prob), time)
#       info("reaction force")
#       dump(reshape(round(la, 8), 2, 6))
    end
end

