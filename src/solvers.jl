# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# Solver stuff

abstract Solver

"""
Solve field equations for a single problem with some dofs fixed. This can be used
to test nonlinear element formulations. Dirichlet boundary is assumed to be homogeneous
and degrees of freedom are eliminated. So if boundary condition is known in nodal
points and everything is zero this should be quite good.
"""
function solve!(problem::Problem, free_dofs::Vector{Int}, time::Float64; max_iterations::Int=10, tolerance::Float64=1.0e-12, dump_matrices::Bool=false, callback=nothing)
    info("start solver")
    assembly = Assembly()
#   x = zeros(ga.ndofs)
#   dx = fill!(similar(x), 0.0)
#   FIXME: better.
    x = nothing
    dx = nothing
    field_name = get_unknown_field_name(problem)
    dim = get_unknown_field_dimension(problem)
    for i=1:max_iterations
        assemble!(assembly, problem, time)
        A = sparse(assembly.stiffness_matrix)
        b = sparse(assembly.force_vector)
        if dump_matrices
            dump(full(A))
            dump(full(b)')
        end
        if isa(dx, Void)
            x = zeros(length(b))
            dx = zeros(length(b))
        end
        dx[free_dofs] = lufact(A[free_dofs,free_dofs]) \ full(b)[free_dofs]
        info("Difference in solution norm: $(norm(dx))")
        x += dx
        if !(isa(callback, Void))
            callback(x)
        end
        for element in get_elements(problem)
            gdofs = get_gdofs(element, problem.dim)
            data = full(x[gdofs])
            if length(data) != length(element)
                data = reshape(data, problem.dim, length(element))
                data = [data[:,i] for i=1:size(data,2)]
            end
            push!(element[field_name], time => data)
        end
        norm(dx) < tolerance && return
    end
    error("Did not converge in $max_iterations iterations")
end


""" Simple linear solver for educational purposes. """
type LinearSolver <: Solver
    field_problem :: Problem
    boundary_problem :: BoundaryProblem
end

"""
Call solver to solve a set of problems.

This is a simple direct solver for demonstration purposes. It handles the
common situation, i.e., some main field problem and it's Dirichlet boundary.

    Ku + C'λ = f
    Cu       = g

"""
function call(solver::LinearSolver, time::Float64)

    field_name = get_unknown_field_name(solver.field_problem)
    field_dim = get_unknown_field_dimension(solver.field_problem)
    info("solving $field_name problem, $field_dim dofs / nodes")

    field_assembly = assemble(solver.field_problem, time)
    boundary_assembly = assemble(solver.boundary_problem, time)

    info("Creating sparse matrices")
    K = sparse(field_assembly.stiffness_matrix)
    dim = size(K, 1)
    f = sparse(field_assembly.force_vector, dim, 1)

    C = sparse(boundary_assembly.stiffness_matrix, dim, dim)
    g = sparse(boundary_assembly.force_vector, dim, 1)

    # create a saddle point problem
    A = [K C'; C' zeros(C)]
    b = [f; g]

    # solve problem
    nz = unique(rowvals(A))  # take only non-zero rows
    x = zeros(b)
    x[nz] = lufact(A[nz,nz]) \ full(b[nz])

    # get "problem-wise" solution vectors
    u = x[1:dim]
    la = x[dim+1:end]

    # update field for elements in problem 1
    for element in get_elements(solver.field_problem)
        gdofs = get_gdofs(element, field_dim)
        local_sol = vec(full(u[gdofs]))
        # if solving vector field, modify local solution vector
        # to array of vectors
        if field_dim != 1
            local_sol = reshape(local_sol, field_dim, length(element))
            local_sol = [local_sol[:,i] for i=1:size(local_sol,2)]
        end
        if haskey(element, field_name)
            push!(element[field_name], time => local_sol)
        else
            element[field_name] = (time => local_sol)
        end
    end

    return norm(u)
end

