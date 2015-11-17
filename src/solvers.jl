# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# Solver stuff

abstract Solver

"""
Solve field equations for single element with some dofs fixed. This can be used
to test nonlinear element formulations.
"""
function solve!(equation::Equation, free_dofs::Vector{Int}, time::Number=0.0; 
                 max_iterations::Int=10, tolerance::Float64=1.0e-12, dump_matrices::Bool=false)
    unknown_field_name = get_unknown_field_name(equation)
    element = get_element(equation)
    x0 = element[unknown_field_name](0.0)
    x = zeros(prod(size(equation)))
    dx = fill!(similar(x), 0.0)
    ass = Assembly()
    for i=1:max_iterations
        empty!(ass)
        assemble!(ass, equation)
        A = full(ass.stiffness_matrix)[free_dofs, free_dofs]
        b = full(ass.force_vector)[free_dofs]
        if dump_matrices
            dump(full(A))
            dump(full(b)')
        end
        dx[free_dofs] = A \ b
        x += dx
        push!(element[unknown_field_name], reshape(x, size(equation)))
        norm(dx) < tolerance && return
    end
    error("Did not converge in $max_iterations iterations")
end

"""
Solve field equations for a single problem with some dofs fixed. This can be used
to test nonlinear element formulations. Dirichlet boundary is assumed to be homogeneous
and degrees of freedom are eliminated. So if boundary condition is known in nodal
points and everything is zero this should be quite good.
"""
function solve!(problem::Problem, free_dofs::Vector{Int}, time::Number=1.0; max_iterations::Int=10, tolerance::Float64=1.0e-12, dump_matrices::Bool=false)
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
        for equation in get_equations(problem)
            element = get_element(equation)
            gdofs = get_gdofs(equation)
            data = reshape(full(x[gdofs]), size(equation))
            push!(element[field_name], data)
        end
        norm(dx) < tolerance && return
    end
    error("Did not converge in $max_iterations iterations")
end

""" Add new problem to solver. """
function add_problem!(solver::Solver, problem::Problem)
    push!(solver.problems, problem)
end

function Base.push!(solver::Solver, problem::Problem)
    push!(solver.problems, problem)
end

""" Get all problems assigned to solver. """
function get_problems(s::Solver)
    return s.problems
end

## SimpleSolver -- tiny direct demo solver
""" Simple solver for educational purposes. """
type SimpleSolver <: Solver
    problems
end

""" Default initializer. """
function SimpleSolver()
    SimpleSolver(Problem[])
end

"""
Call solver to solve a set of problems.

This is a simple direct solver for demonstration purposes. It handles the
common situation, i.e., some main field problem and it's Dirichlet boundary.

    Au + C'λ = f
    Cu       = g

"""
function call(solver::SimpleSolver, time::Number=0.0)
    problem1, problem2 = get_problems(solver)

    assembly1 = Assembly()
    assemble!(assembly1, problem1, time)
    assembly2 = Assembly()
    assemble!(assembly2, problem2, time)

#   info("Creating sparse matrices")
    A1 = sparse(assembly1.stiffness_matrix)
    b1 = sparse(assembly1.force_vector, size(A1, 1), 1)
    A2 = sparse(assembly2.stiffness_matrix)
    b2 = sparse(assembly2.force_vector, size(A2, 1), 1)
    
    # create a saddle point problem
    A = [A1 A2; A2' zeros(A2)]
    b = [b1; b2]

    # solve problem
    nz = unique(rowvals(A))  # take only non-zero rows
    x = zeros(b)
    x[nz] = lufact(A[nz,nz]) \ full(b[nz])

    # get "problem-wise" solution vectors
    x1 = x[1:length(b1)]
    x2 = x[length(b1)+1:end]

    # update field for elements in problem 1
    for equation in get_equations(problem1)
        element = get_element(equation)
        field_name = get_unknown_field_name(problem1)
        gdofs = get_gdofs(problem1, equation)
        element_solution = full(x1[gdofs])
        field = Increment(element_solution)
        push!(element[field_name], TimeStep(time, field))
    end

    # update field for elements in problem 2 (Dirichlet boundary)
    for equation in get_equations(problem2)
        element = get_element(equation)
        field_name = get_unknown_field_name(problem2)
        gdofs = get_gdofs(problem2, equation)
        element_solution = full(x2[gdofs])
        field = Increment(element_solution)
        push!(element[field_name], TimeStep(time, field))
    end
end

