# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# Solver stuff

abstract Solver

""" Add new problem to solver. """
function add_problem!(solver::Solver, problem::Problem)
    push!(solver.problems, problem)
end
function Base.push!(solver::Solver, problem::Problem)
    push!(solver.problems, problem)
end

"""
Get all problems assigned to solver
"""
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

This is simple serial solver for demonstration purposes. It handles the most
common situation, i.e., some main field problem and it's Dirichlet boundary.
"""
function call(solver::SimpleSolver, t)
    problems = get_problems(solver)
    problem1 = problems[1]
    problem2 = problems[2]

    # calculate order of degrees of freedom in global matrix
    # and set the ordering to problems
    dofmap = calculate_global_dofs(problem1)
    assign_global_dofs!(problem1, dofmap)
    assign_global_dofs!(problem2, dofmap)

    # assemble problem 1
    A1 = sparse(get_lhs(problem1, t)...)
    b1 = sparsevec(get_rhs(problem1, t)..., size(A1, 1))

    # assemble problem 2
    A2 = sparse(get_lhs(problem2, t)...)
    b2 = sparsevec(get_rhs(problem2, t)..., size(A2, 1))

    # make one monolithic assembly
    A = [A1 A2; A2' zeros(A2)]
    b = [b1; b2]

    # solve problem
    nz = unique(rowvals(A))
    x = zeros(b)
    x[nz] = lufact(A[nz,nz]) \ full(b[nz])

    # get "problem-wise" solution vectors
    x1 = x[1:length(b1)]
    x2 = x[length(b1)+1:end]

    # check residual
    R1 = A1*x1 - b1
    R2 = A2*x2 - b2
    println("Residual norm: $(norm(R1+R2))")

    # update field for elements in problem 1
    for equation in get_equations(problem1)
        gdofs = get_global_dofs(equation)
        element = get_element(equation)
        field_name = get_unknown_field_name(equation)  # field we are solving
        field = Field(t, full(x1[gdofs])[:])
        push!(element[field_name], field)
    end

    # update field for elements in problem 2
    for equation in get_equations(problem2)
        gdofs = get_global_dofs(equation)
        element = get_element(equation)
        field_name = get_unknown_field_name(equation)
        field = Field(t, full(x2[gdofs]))
        push!(element[field_name], field)
    end
end

