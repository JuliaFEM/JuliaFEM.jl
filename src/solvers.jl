# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

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
type LinearSolver
    name :: ASCIIString
    field_problems :: Vector{Problem}
    boundary_problems :: Vector{BoundaryProblem}
end

function LinearSolver(name="LinearSolver")
    LinearSolver(name, [], [])
end

function push!(solver::LinearSolver, problem::Problem)
    length(solver.field_problems) == 0 || error("Only one field problem allowed for LinearSolver")
    push!(solver.field_problems, problem)
end

function push!(solver::LinearSolver, problem::BoundaryProblem)
    length(solver.boundary_problems) == 0 || error("Only one boundary problem allowed for LinearSolver")
    push!(solver.boundary_problems, problem)
end

"""
Call solver to solve a set of problems.

This is a simple direct solver for demonstration purposes. It handles the
common situation, i.e., some main field problem and it's Dirichlet boundary.

    Ku + C'λ = f
    Cu       = g

"""
function call(solver::LinearSolver, time::Float64)

    t0 = Base.time()
    field_name = get_unknown_field_name(solver.field_problems[1])
    field_dim = get_unknown_field_dimension(solver.field_problems[1])
    info("solving $field_name problem, $field_dim dofs / nodes")

    field_assembly = assemble(solver.field_problems[1], time)
    boundary_assembly = assemble(solver.boundary_problems[1], time)

    #info("Creating sparse matrices")
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
    for element in get_elements(solver.field_problems[1])
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

    t1 = round(Base.time()-t0, 2)
    info("solved problem in $t1 seconds.")
    return norm(u)
end


# Tuple{Symbol,Any,Any} or Function

type Solver
    name :: ASCIIString
    time :: Real
    iteration :: Int
    problems :: Vector{Union{FieldProblem, BoundaryProblem}}
    is_linear_system :: Bool
    nonlinear_system_max_iterations :: Int64
    nonlinear_system_convergence_tolerance :: Float64
    linear_system_solver :: Symbol
end

function Solver(name::ASCIIString="default solver", time::Real=0.0)
    return Solver(
        name,  # name
        time,  # time
        0,     # iteration counter
        [],    # array of problems
        false, # is this a linear system which can be solved in a single iteration?
        10,    # max nonlinear iterations
        5.0e-5, # nonlinear iteration convergence tolerance
        :DirectLinearSolver # linear system solution method
        )
end

function push!(solver::Solver, problem::Union{FieldProblem, BoundaryProblem})
    push!(solver.problems, problem)
end

# one-liner helpers to identify problem types

function is_field_problem(problem)
    return typeof(problem) <: FieldProblem
end

function is_boundary_problem(problem)
    return typeof(problem) <: BoundaryProblem
end

function is_dirichlet_problem(problem)
    return typeof(problem) <: Union{BoundaryProblem{DirichletProblem}, BoundaryProblem{DirichletProblem{DualBasis}}}
end

function is_mortar_problem(problem)
    return typeof(problem) <: BoundaryProblem{MortarProblem}
end

function get_field_problems(solver::Solver)
    filter(is_field_problem, solver.problems)
end

function get_boundary_problems(solver::Solver)
    filter(is_boundary_problem, solver.problems)
end

function get_dirichlet_problems(solver::Solver)
    filter(is_dirichlet_problem, solver.problems)
end

function get_mortar_problems(solver::Solver)
    filter(is_mortar_problem, solver.problems)
end

"""Return one combined field assembly for a set of field problems.

Parameters
----------
solver :: Solver

Returns
-------
K, f :: SparseMatrixCOO

Notes
-----
If several field problems exists, they are simply summed together, so
problems must have unique node ids.

"""
function get_field_assembly(solver::Solver)
    return get_field_assembly(get_field_problems(solver))
end
function get_field_assembly(problems::Vector{Union{BoundaryProblem, FieldProblem}})
    K = SparseMatrixCOO()
    f = SparseMatrixCOO()
    for problem in problems
        append!(K, problem.assembly.stiffness_matrix)
        append!(f, problem.assembly.force_vector)
    end
    return K, f
end

""" Return one combined boundary assembly for a set of boundary problems.

Returns
-------
C1, C2, D, g :: SparseMatrixCOO

"""
function get_boundary_assembly(solver::Solver)
    return get_boundary_assembly(get_boundary_problems(solver))
end
function get_boundary_assembly(problems::Vector{Union{BoundaryProblem, FieldProblem}})
    C1 = SparseMatrixCOO()
    C2 = SparseMatrixCOO()
    D = SparseMatrixCOO()
    g = SparseMatrixCOO()
    for problem in problems
        append!(C1, problem.assembly.C1)
        append!(C2, problem.assembly.C2)
        append!(D, problem.assembly.D)
        append!(g, problem.assembly.g)
    end
    return C1, C2, D, g
end

""" Solve linear system using LU factorization (UMFPACK).
"""
function solve_linear_system!(solver::Solver, ::Type{Val{:DirectLinearSolver}})
    info("solving linear system of $(length(solver.problems)) problems.")
    t0 = time()

    # assemble field problems
    K, f = get_field_assembly(solver)
    K = sparse(K)
    dim = size(K, 1)
    f = sparse(f, dim, 1)

    # assemble boundary problems
    C1, C2, D, g = get_boundary_assembly(solver)
    C1 = sparse(C1, dim, dim)
    C2 = sparse(C2, dim, dim)
    D = sparse(D, dim, dim)
    g = sparse(g, dim, 1)

    # construct global system Ax=b and solve using lu factorization
    A = [K C1'; C2 D]
    b = [f; g]
    nz1 = sort(unique(rowvals(A)))
    nz2 = sort(unique(rowvals(A')))
    x = zeros(length(b))
    x[nz1] = lufact(A[nz1,nz2]) \ full(b[nz1])

    # update solutions
    u = x[1:dim]
    la = x[dim+1:end]
    for problem in solver.problems
        typeof(problem) <: FieldProblem && update!(problem, u)
        typeof(problem) <: BoundaryProblem && update!(problem, la)
    end

    info("UMFPACK: solved in ", time()-t0, " seconds. norm = ", norm(u))
end

""" Check convergence of problems.

Notes
-----
Default convergence criteria is obtained by checking each sub-problem convergence.
"""
function has_converged(solver::Solver; print_convergence_information=true)
    converged = true
    for problem in solver.problems
        has_converged = problem.assembly.solution_norm_change < solver.nonlinear_system_convergence_tolerance
        if print_convergence_information
            @printf "% 30s | %8.3f | %s\n" problem.name problem.assembly.solution_norm_change has_converged
        end
        converged &= has_converged
    end
    return converged || solver.is_linear_system
end

type NonlinearConvergenceError <: Exception
    solver :: Solver
end

function Base.showerror(io::IO, exception::NonlinearConvergenceError)
    max_iters = exception.solver.nonlinear_system_max_iterations
    print(io, "nonlinear iteration did not converge in $max_iters iterations!")
end

""" Main solver loop.
"""
function call(solver::Solver)
    # 1. initialize each problem so that we can start nonlinear iterations
    for problem in solver.problems
        initialize!(problem, solver.time)
    end

    # 2. start non-linear iterations
    for solver.iteration=1:solver.nonlinear_system_max_iterations
        # 2.1 update linearized assemblies (if needed)
        for problem in solver.problems
            problem.assembly.changed = true  # force reassembly
            assemble!(problem, solver.time)
        end

        # 2.2 call solver for linearized system (default: direct lu factorization)
        solve_linear_system!(solver, Val{solver.linear_system_solver})

        # 2.3 update solution back to elements
        for problem in solver.problems
            update!(problem, problem.assembly.solution, Val{:elements})
        end

        # 2.4 check convergence
        if has_converged(solver)
            info("Converged in $(solver.iteration) iterations.")
            return true
        end
    end

    # 3. did not converge
    throw(NonlinearConvergenceError(solver))
end

