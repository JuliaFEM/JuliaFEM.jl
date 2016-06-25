# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

abstract AbstractSolver

type Solver{S<:AbstractSolver}
    name :: ASCIIString # some descriptive name for problem
    time :: Real        # current time
    problems :: Vector{Problem}
    ndofs :: Int        # total dimension of global stiffness matrix, i.e., dim*nnodes
    properties :: S
end

type Nonlinear <: AbstractSolver
    iteration :: Int    # iteration counter
    norms :: Vector{Tuple} # solution norms for convergence studies
    min_iterations :: Int64
    max_iterations :: Int64
    convergence_tolerance :: Float64
    error_if_no_convergence :: Bool
    is_linear_system :: Bool # setting this to true makes assumption of one step convergence
    linear_system_solver :: Symbol
end

function Nonlinear()
    solver = Nonlinear(
        0,     # iteration number
        [],    # solution norms in (norm(u), norm(la)) tuples
        1,     # min nonlinear iterations
        10,    # max nonlinear iterations
        5.0e-5, # nonlinear iteration convergence tolerance
        true,  # throw error if no convergence
        false, # is_linear_system
        :DirectLinearSolver) # linear system solution method
   return solver
end

function Solver{S<:AbstractSolver}(::Type{S}=Nonlinear,
                                   name::ASCIIString="default solver",
                                   time::Real=0.0, problems=[],
                                   properties...)
    variant = S(properties...)
    solver = Solver{S}(name, time, problems, 0, variant)
    return solver
end

""" For compatibility. """
function Solver(name::ASCIIString="default solver",
                time::Real=0.0, problems=[],
                properties...)
    variant = Nonlinear(properties...)
    solver = Solver{Nonlinear}(name, time, problems, 0, variant)
    return solver
end

function get_problems(solver::Solver)
    return solver.problems
end

function push!(solver::Solver, problem)
    push!(solver.problems, problem)
end

function getindex(solver::Solver, problem_name::ASCIIString)
    for problem in get_problems(solver)
        if problem.name == problem_name
            return problem
        end
    end
    throw(KeyError(problem_name))
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
function get_field_assembly(solver::Solver; symmetric=true,
                            with_mass_matrix=false,
                            empty_after_append=true)
    problems = get_field_problems(solver)
    M = SparseMatrixCOO()
    K = SparseMatrixCOO()
    Kg = SparseMatrixCOO()
    f = SparseMatrixCOO()
    for problem in problems
        append!(K, problem.assembly.K)
        append!(Kg, problem.assembly.Kg)
        append!(f, problem.assembly.f)
        with_mass_matrix && append!(M, problem.assembly.M)
        empty_after_append && empty!(problem.assembly)
    end
    if solver.ndofs == 0
        solver.ndofs = size(K, 1)
    end
    K = sparse(K, solver.ndofs, solver.ndofs)
    Kg = sparse(Kg, solver.ndofs, solver.ndofs)
    M = sparse(M, solver.ndofs, solver.ndofs)
    if symmetric
        K = 1/2*(K + K')
        Kg = 1/2*(Kg + Kg')
        M = 1/2*(M + M')
    end
    f = sparse(f, solver.ndofs, 1)

    # run any posthook for assembly if defined
    args = Tuple{Solver, SparseMatrixCSC, SparseMatrixCSC, SparseMatrixCSC}
    if method_exists(field_assembly_posthook!, args)
        field_assembly_posthook!(solver, K, Kg, f)
    end

    return M, K, Kg, f
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
    K = spzeros(ndofs, ndofs)
    C1 = spzeros(ndofs, ndofs)
    C2 = spzeros(ndofs, ndofs)
    D = spzeros(ndofs, ndofs)
    f = spzeros(ndofs, 1)
    g = spzeros(ndofs, 1)
    for problem in get_boundary_problems(solver)
        assembly = problem.assembly
        K_ = sparse(assembly.K, ndofs, ndofs)
        C1_ = sparse(assembly.C1, ndofs, ndofs)
        C2_ = sparse(assembly.C2, ndofs, ndofs)
        D_ = sparse(assembly.D, ndofs, ndofs)
        f_ = sparse(assembly.f, ndofs, 1)
        g_ = sparse(assembly.g, ndofs, 1)
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
        K += K_
        C1 += C1_
        C2 += C2_
        D += D_
        f += f_
        g += g_
        empty!(problem.assembly)
    end
    return K, C1, C2, D, f, g
end


"""
Construct new basis such that u = P*uh + g

Parameters
----------
S  set of linearly independent dofs.
"""
function create_projection(C::SparseMatrixCSC, g; S=nothing, tol=1.0e-12)
    n, m = size(C)
    @assert n == m
    if S == nothing
        S = get_nonzero_rows(C)
    end
    # FIXME: this creates dense matrices
    # efficiency / memory usage is a question
    M = get_nonzero_columns(C)
    F = qrfact(C[S,:])
    P = spzeros(n,m)
    P[:,M] = sparse(F \ full(C[S,M]))
    h = sparse(F \ full(g[S]))
    resize!(P, n, m)
    resize!(h, n, 1)
    P = speye(n) - P
    SparseMatrix.droptol!(P, tol)
    return P, h
end


"""
Solve linear system using LDLt factorization (SuiteSparse). This version
requires that final system is symmetric and positive definite, so boundary
conditions are first eliminated before solution.
"""
function solve!(K, C1, C2, D, f, g, u, la, ::Type{Val{1}}; debug=false)

    nnz(D) == 0 || return false
    nz = get_nonzero_rows(C2)
    B = get_nonzero_rows(C2')
    # C2^-1 exists or this doesn't work
    length(nz) == length(B) || return false

    A = get_nonzero_rows(K)
    I = setdiff(A, B)

    if debug
        info("# nz = $(length(nz))")
        info("# A = $(length(A))")
        info("# B = $(length(B))")
        info("# I = $(length(I))")
    end

    # solver boundary dofs
    try
        u[B] = lufact(C2[nz,B]) \ full(g[nz])
    catch
        info("solver #1 failed to solve boundary dofs (you should not see this message).")
        return false
    end

    # solve interior domain using LDLt factorization
    u[I] = ldltfact(K[I,I]) \ (f[I] - K[I,B]*u[B])
    # solve lambda
    la[B] = lufact(C1[B,nz]) \ full(f[B] - K[B,I]*u[I] - K[B,B]*u[B])

    return true
end

"""
Solve linear system using LU factorization (UMFPACK). This version solves
directly the saddle point problem without elimination of boundary conditions.
"""
function solve!(K, C1, C2, D, f, g, u, la, ::Type{Val{2}})
    # construct global system Ax = b and solve using lufact (UMFPACK)
    A = [K C1'; C2  D]
    b = [f; g]
    nz = get_nonzero_rows(A)
    x = zeros(length(b))
    x[nz] = lufact(A[nz,nz]) \ full(b[nz])
    ndofs = size(K, 1)
    u[:] = x[1:ndofs]
    la[:] = x[ndofs+1:end]
    return true
end

function solve_linear_system(solver::Solver)
    info("solving linear system of $(length(solver.problems)) problems.")
    t0 = time()

    # assemble field problems
    M, K, Kg, f = get_field_assembly(solver)
    # assemble boundary problems
    Kb, C1, C2, D, fb, g = get_boundary_assembly(solver)
    K = K + Kg + Kb
    K = 1/2*(K + K')
    f = f + fb

    u = zeros(solver.ndofs)
    la = zeros(solver.ndofs)

    status = false
    for i in [1, 2]
        status = solve!(K, C1, C2, D, f, g, u, la, Val{i})
        if status
            info("succesfully solved Ax = b using solver #$i")
            break
        end
    end
    status || error("Failed to solve linear system!")

    info("linear system solver: solved in ", time()-t0, " seconds. norm = ", norm(u))
    return u, la
end

""" Check convergence of problems.

Notes
-----
Default convergence criteria is obtained by checking each sub-problem convergence.
"""
function has_converged(solver::Solver{Nonlinear};
                       check_convergence_for_boundary_problems=false)
    properties = solver.properties
    converged = true
    eps = properties.convergence_tolerance
    for problem in solver.problems
        has_converged = true
        if is_field_problem(problem)
            has_converged = problem.assembly.u_norm_change < eps
            if isapprox(norm(problem.assembly.u), 0.0)
                has_converged = true
            end
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
    return converged || properties.is_linear_system
end

type NonlinearConvergenceError <: Exception
    solver :: Solver
end

function Base.showerror(io::IO, exception::NonlinearConvergenceError)
    max_iters = exception.solver.properties.max_iterations
    print(io, "nonlinear iteration did not converge in $max_iters iterations!")
end

function assemble!(solver::Solver; force_assembly=true)
    info("Assembling problems ...")
    tic()
    for problem in solver.problems
        if force_assembly # force reassembly
            problem.assembly.changed = true
        end
        assemble!(problem, solver.time)
    end
    t1 = round(toq(), 2)
    info("Assembled in $t1 seconds.")
end

function initialize!(solver::Solver)
    for problem in solver.problems
        initialize!(problem, solver.time)
    end
end

""" Default solver for quasistatic nonlinear problems. """
function call(solver::Solver{Nonlinear})

    properties = solver.properties

    # 1. initialize each problem so that we can start nonlinear iterations
    initialize!(solver)

    # 2. start non-linear iterations
    for properties.iteration=1:properties.max_iterations
        info("Starting nonlinear iteration #$(properties.iteration)")

        # 2.1 update linearized assemblies (if needed)
        assemble!(solver)

        # 2.2 call solver for linearized system (default: direct lu factorization)
        info("Solve linear system ...")
        tic()
        u, la = solve_linear_system(solver)
        push!(properties.norms, (norm(u), norm(la)))
        t1 = round(toq(), 2)
        info("Solved Ax = b in $t1 seconds.")

        # 2.3 update solution back to elements
        for problem in solver.problems
            u_new, la_new = update_assembly!(problem, u, la)
            update_elements!(problem, u_new, la_new)
        end

        # 2.4 check convergence
        if has_converged(solver)
            info("Converged in $(properties.iteration) iterations.")
            if properties.iteration < properties.min_iterations
                info("Converged but continuing")
            else
                return true
            end
        end
    end

    # 3. did not converge
    if properties.error_if_no_convergence
        throw(NonlinearConvergenceError(solver))
    end
end

