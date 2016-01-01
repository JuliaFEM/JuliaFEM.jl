# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM.Test

using JuliaFEM.Core: Node, Seg2, Tri3, update!, calculate_normal_tangential_coordinates!,
                     PlaneStressLinearElasticityProblem, DirichletProblem, MortarProblem,
                     get_elements, DirectSolver, calculate_nodal_vector, FieldAssembly,
                     FieldProblem, set_linear_system_solver!, set_nonlinear_max_iterations!,
                     get_elements, Element, BoundaryAssembly, BoundaryProblem,
                     get_integration_points, get_jacobian, get_connectivity, StandardBasis,
                     add_postprocessor!, add_preprocessor!, SparseMatrixCOO, add!,
                     add_linear_system_solver_preprocessor!,
                     add_linear_system_solver_postprocessor!

import JuliaFEM.Core: assemble_preprocess!, assemble_postprocess!,
                      linear_system_solver_preprocess!, linear_system_solver_postprocess!

macro debug(msg)
    haskey(ENV, "DEBUG") || return
    return msg
end

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
    return SparseMatrixCOO(P)
end

function assemble_postprocess!(assembly, problem, time::Real, ::Type{Val{:remove_constraint_from_dofs}}, dofs)
    # giving additional arguments and keywords is possible too
    C1 = sparse(assembly.C1)
    C2 = sparse(assembly.C2)
    info("removing constraints from dofs: $dofs")
    for d in dofs
        C1[d,:] = 0
        C2[d,:] = 0
    end
    assembly.C1 = C1
    assembly.C2 = C2
end

function assemble_postprocess!(assembly, problem, time::Real, ::Type{Val{:remove_tangential_constraints}})
    # example how to use postprocessor to manipulate constraint matrix before summing assemblies together
    info("postprocess mortar assembly: remove contraints in tangent direction on boundary.")
    dim = 12
    C1 = sparse(assembly.C1, dim, dim)
    C2 = sparse(assembly.C2, dim, dim)
    P = calculate_normal_tangential_coordinates(get_elements(problem), time)
    P = sparse(P, dim, dim)
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

function assemble_postprocess!(assembly, problem, time::Real, ::Type{Val{:primal_dual_active_set_strategy}})
    info("PDASS: determining active contact set")
    dim = 12
    C1 = sparse(assembly.C1, dim, dim)
    C2 = sparse(assembly.C2, dim, dim)
    elements = get_elements(problem)
    P = calculate_normal_tangential_coordinates(get_elements(problem), time)
    P = sparse(P, dim, dim)

    la = calculate_nodal_vector("reaction force", 2, elements, time)
    X = calculate_nodal_vector("geometry", 2, elements, time)
    u = calculate_nodal_vector("displacement", 2, elements, time)
    resize!(la, 12)
    resize!(X, 12)
    resize!(u, 12)
    x = X+u
    info("x = \n", reshape(round(x, 2), 2, 6))
    P = sparse(eye(dim))
    C1 = P*C1
    C2 = P*C2
    gn = P*C1*X
    un = P*C1*u
    la = P*la
    info("weighted gap in nt =")
    dump(reshape(round(gn+un, 2), 2, 6))
    info("lambda in nt =")
    dump(reshape(round(la, 2), 2, 6))
    # complementarity function
    cn = 1.0
    C = la - clamp(la - cn*(gn+un), 0, Inf)
    info("complementarity function =")
    dump(reshape(round(C, 2), 2, 6))

    g = zeros(length(gn))

    for i=1:2:dim
        #if C[i] < 0
        if i==1
            info("dof $i in active set")
            g[i] = -gn[i]
        else
            info("dof $i not in active set")
            C1[i,:] = 0
            C2[i,:] = 0
        end
    end

    for i=2:2:dim
        C1[i,:] = 0
        C2[i,:] = 0
    end

    assembly.g = sparse(g)
    assembly.C1 = C1
    assembly.C2 = C2
    info("PDASS ready.")
end

function linear_system_solver_preprocess!(solver, iter, time, K, f, C1, C2, D, g, sol, la, ::Type{Val{:before_solution}})
    # example how to use preprocessor to dump matrices before solution
    @debug begin
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
end

function linear_system_solver_postprocess!(solver, iter, time, K, f, C1, C2, D, g, x, la, ::Type{Val{:after_solution}})
    @debug begin
        info("solution vector")
        dump(round(full(x)', 3))
        info("reaction force vector")
        dump(round(full(la)', 3))
    end
end

@testset "2d frictionless contact" begin
    gap = [1.0, 0.0]
    nodes = Node[
        [6.0, 6.0],
        [6.0, 12.0]+gap,
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
    update!([force], "displacement traction force 1", 2*6/sqrt(2))

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
    # add_postprocessor!(cont, :remove_constraint_from_dofs, [7, 8])
    # remove tangential direction constraints
    # add_postprocessor!(cont, :remove_tangential_constraints)
    # apply PDASS
    add_postprocessor!(cont, :primal_dual_active_set_strategy)

    @debug begin
        info("fel1.fields = $(fel1.fields)")
    end

    solver = DirectSolver()
    push!(solver, prob)
    push!(solver, bc)
    push!(solver, cont)
    solver.solve_residual = false
    set_linear_system_solver!(solver, :UMFPACK)
    set_nonlinear_max_iterations!(solver, 5)
    add_linear_system_solver_preprocessor!(solver, :before_solution)
    add_linear_system_solver_postprocessor!(solver, :after_solution)
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

