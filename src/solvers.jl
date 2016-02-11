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
    boundary_problems :: Vector{Problem}
end

function LinearSolver(name="LinearSolver")
    LinearSolver(name, [], [])
end

function push!{P<:FieldProblem}(solver::LinearSolver, problem::Problem{P})
    length(solver.field_problems) == 0 || error("Only one field problem allowed for LinearSolver")
    push!(solver.field_problems, problem)
end

function push!{P<:BoundaryProblem}(solver::LinearSolver, problem::Problem{P})
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
    name :: ASCIIString # some descriptive name for problem
    time :: Real        # current time
    iteration :: Int    # iteration counter
    ndofs :: Int        # total dimension of global stiffness matrix, i.e., dim*nnodes
    problems :: Vector{Problem}
    is_linear_system :: Bool # setting this to true makes assumption of one step convergence
    nonlinear_system_max_iterations :: Int64
    nonlinear_system_convergence_tolerance :: Float64
    linear_system_solver :: Symbol
end

function Solver(name::ASCIIString="default solver", time::Real=0.0)
    return Solver(
        name,
        time,
        0,     # iteration #
        0,     # ndofs
        [],    # array of problems
        false, # is_linear_system
        10,    # max nonlinear iterations
        5.0e-5, # nonlinear iteration convergence tolerance
        :DirectLinearSolver # linear system solution method
        )
end

function push!(solver::Solver, problem)
    push!(solver.problems, problem)
end

# one-liner helpers to identify problem types

function is_field_problem(problem)
    return false
end
function is_field_problem{P<:FieldProblem}(problem::Problem{P})
    return true
end

function is_boundary_problem(problem)
    return false
end
function is_boundary_problem{P<:BoundaryProblem}(problem::Problem{P})
    return true
end

function is_dirichlet_problem(problem)
    return false
end
function is_dirichlet_problem{P<:Problem{Dirichlet}}(problem::P)
    return true
end

#=
function is_mortar_problem{P<:Problem{Mortar}}(problem::P)
    return true
end
=#

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

""" Posthook for field assembly. By default, do nothing. """
function field_assembly_posthook!
end

"""Return one combined field assembly for a set of field problems.

Parameters
----------
solver :: Solver

Returns
-------
K, f :: SparseMatrixCSC

Notes
-----
If several field problems exists, they are simply summed together, so
problems must have unique node ids.

"""
function get_field_assembly(solver::Solver)
    problems = get_field_problems(solver)
    K = SparseMatrixCOO()
    f = SparseMatrixCOO()
    for problem in problems
        assembly = problem.assembly
        append!(K, assembly.K)
        append!(f, assembly.f)
    end
    K = sparse(K)
    solver.ndofs = size(K, 1)
    f = sparse(f, solver.ndofs, 1)

    # run any posthook for assembly if defined
    args = Tuple{Solver, SparseMatrixCSC, SparseMatrixCSC}
    if method_exists(field_assembly_posthook!, args)
        field_assembly_posthook!(solver, K, f)
    end

    return K, f
end

""" Posthook for boundary assembly. By default, do nothing. """
function boundary_assembly_posthook!
end

""" Return one combined boundary assembly for a set of boundary problems.

Returns
-------
C1, C2, D, g :: SparseMatrixCSC

Notes
-----
When some dof is constrained by multiple boundary problems an algorithm is
launched what tries to do it's best to solve issue. It's far from perfect
but is able to handle some basic situations occurring in corner nodes and
crosspoints.

"""
function get_boundary_assembly(solver::Solver)
    ndofs = solver.ndofs
    @assert ndofs != 0
    C1 = spzeros(ndofs, ndofs)
    C2 = spzeros(ndofs, ndofs)
    D = spzeros(ndofs, ndofs)
    g = spzeros(ndofs, 1)
    for problem in get_boundary_problems(solver)
        assembly = problem.assembly
        C1_ = sparse(assembly.C1, ndofs, ndofs)
        C2_ = sparse(assembly.C2, ndofs, ndofs)
        D_ = sparse(assembly.D, ndofs, ndofs)
        g_ = sparse(assembly.g, ndofs, 1)

        # boundary assembly posthook: if boundary assembly needs some further
        # manipulations before adding it to global constraint matrix, i.e.,
        # remove some constraints based on some conditions etc. do it here
        args = Tuple{Solver, typeof(problem), SparseMatrixCSC, SparseMatrixCSC,
                     SparseMatrixCSC, SparseMatrixCSC}
        if method_exists(boundary_assembly_posthook!, args)
            boundary_assembly_posthook!(solver, problem, C1_, C2_, D_, g_)
        end

        # check for overconstraint situation and handle it if possible
        already_constrained = get_nonzero_rows(C2)
        new_constraints = get_nonzero_rows(C2_)
        overconstrained_dofs = intersect(already_constrained, new_constraints)
        if length(overconstrained_dofs) != 0
            overconstrained_dofs = sort(overconstrained_dofs)
            overconstrained_nodes = find_nodes_by_dofs(problem, overconstrained_dofs)
            handle_overconstraint_error!(problem, overconstrained_nodes,
                overconstrained_dofs, C1, C1_, C2, C2_, D, D_, g, g_)
        end
        C1 += C1_
        C2 += C2_
        D += D_
        g += g_
    end
    return C1, C2, D, g
end


""" Solve linear system using LU factorization (UMFPACK).
"""
function solve_linear_system(solver::Solver, ::Type{Val{:DirectLinearSolver}})
    info("solving linear system of $(length(solver.problems)) problems.")
    t0 = time()

    # assemble field problems
    K, f = get_field_assembly(solver)

    # assemble boundary problems
    C1, C2, D, g = get_boundary_assembly(solver)

    # construct global system Ax=b and solve using lu factorization
    A = [K C1'; C2 D]
    b = [f; g]

    nz = get_nonzero_rows(A)
    x = zeros(length(b))
    x[nz] = lufact(A[nz,nz]) \ full(b[nz])

    ndofs = solver.ndofs
    u = x[1:ndofs]
    la = x[ndofs+1:end]
    info("UMFPACK: solved in ", time()-t0, " seconds. norm = ", norm(u))
    return u, la
end


""" Check convergence of problems.

Notes
-----
Default convergence criteria is obtained by checking each sub-problem convergence.
"""
function has_converged(solver::Solver; check_convergence_for_boundary_problems=false)
    converged = true
    eps = solver.nonlinear_system_convergence_tolerance
    for problem in solver.problems
        has_converged = true
        if is_field_problem(problem)
            has_converged = problem.assembly.u_norm_change/norm(problem.assembly.u) < eps
            info("Details for problem $(problem.name)")
            info("Norm: $(norm(problem.assembly.u))")
            info("Norm change: $(problem.assembly.u_norm_change)")
            info("Has converged? $(has_converged)")
        end
        if is_boundary_problem(problem) && check_convergence_for_boundary_problems
            has_converged = problem.assembly.la_norm_change/norm(problem.assembly.la) < eps
            info("Details for problem $(problem.name)")
            info("Norm: $(norm(problem.assembly.la))")
            info("Norm change: $(problem.assembly.la_norm_change)")
            info("Has converged? $(has_converged)")
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
        u, la = solve_linear_system(solver, Val{solver.linear_system_solver})

        # 2.3 update solution back to elements
        for problem in solver.problems
            u, la = update_assembly!(problem, u, la)
            update_elements!(problem, u, la)
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
