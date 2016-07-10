# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

abstract AbstractSolver

type Solver{S<:AbstractSolver}
    name :: AbstractString # some descriptive name for problem
    time :: Float64        # current time
    problems :: Vector{Problem}
    norms :: Vector{Tuple}  # solution norms for convergence studies
    ndofs :: Int        # number of degrees of freedom in problem
    properties :: S
end


function Solver{S<:AbstractSolver}(::Type{S}, name="solver", properties...)
    variant = S(properties...)
    solver = Solver{S}(name, 0.0, [], [], 0, variant)
    return solver
end

function get_problems(solver::Solver)
    return solver.problems
end

function push!(solver::Solver, problem)
    push!(solver.problems, problem)
end

function getindex(solver::Solver, problem_name)
    for problem in get_problems(solver)
        if problem.name == problem_name
            return problem
        end
    end
    throw(KeyError(problem_name))
end

# one-liner helpers to identify problem types

is_field_problem(problem) = false
is_field_problem{P<:FieldProblem}(problem::Problem{P}) = true
is_boundary_problem(problem) = false
is_boundary_problem{P<:BoundaryProblem}(problem::Problem{P}) = true
get_field_problems(solver::Solver) = filter(is_field_problem, get_problems(solver))
get_boundary_problems(solver::Solver) = filter(is_boundary_problem, get_problems(solver))

"""
Posthook for field assembly. By default, do nothing.
This can be used to make some modifications for assembly
after all elements are assembled.

Examples
--------
function field_assembly_posthook!(solver::Solver,
                                  K::SparseMatrixCSC,
                                  Kg::SparseMatrixCSC,
                                  f::SparseMatrixCSC,
                                  fg::SpareMatrixCSC)
    info("doing stuff, size(K) = ", size(K))
end
"""
function field_assembly_posthook!
end

"""Return one combined field assembly for a set of field problems.

Parameters
----------
solver :: Solver

Returns
-------
M, K, Kg, f, fg :: SparseMatrixCSC

Notes
-----
If several field problems exists, they are simply summed together, so
problems must have unique node ids.

"""
function get_field_assembly(solver::Solver; show_info=true)
    problems = get_field_problems(solver)

    M = SparseMatrixCOO()
    K = SparseMatrixCOO()
    Kg = SparseMatrixCOO()
    f = SparseMatrixCOO()
    fg = SparseMatrixCOO()

    for problem in problems
        append!(M, problem.assembly.M)
        append!(K, problem.assembly.K)
        append!(Kg, problem.assembly.Kg)
        append!(f, problem.assembly.f)
        append!(fg, problem.assembly.fg)
    end

    if solver.ndofs == 0
        solver.ndofs = size(K, 1)
        show_info && info("automatically determined problem dimension, ndofs = $(solver.ndofs)")
    end

    M = sparse(M, solver.ndofs, solver.ndofs)
    K = sparse(K, solver.ndofs, solver.ndofs)
    if nnz(K) == 0
        warn("Field assembly seems to be empty. Check that elements are pushed to problem and formulation is correct.")
    end
    Kg = sparse(Kg, solver.ndofs, solver.ndofs)
    f = sparse(f, solver.ndofs, 1)
    fg = sparse(fg, solver.ndofs, 1)

    # run any posthook for assembly if defined
    args = Tuple{Solver, SparseMatrixCSC, SparseMatrixCSC, SparseMatrixCSC, SparseMatrixCSC}
    if method_exists(field_assembly_posthook!, args)
        field_assembly_posthook!(solver, K, Kg, fg, fg)
    end

    return M, K, Kg, f, fg
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
    end
    return K, C1, C2, D, f, g
end

function resize!(A::SparseMatrixCSC, m::Int64, n::Int64)
    (n == A.n) && (m == A.m) && return
    @assert n >= A.n
    @assert m >= A.m
    append!(A.colptr, A.colptr[end]*ones(Int, m-A.m))
    A.n = n
    A.m = m
end

"""
Given C and g, construct new basis such that v = P*u + g

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
function solve!(K, C1, C2, D, f, g, u, la, ::Type{Val{1}}; F=nothing, debug=false)

    nnz(D) == 0 || return F, false
    nz = get_nonzero_rows(C2)
    B = get_nonzero_rows(C2')
    # C2^-1 exists or this doesn't work
    length(nz) == length(B) || return F, false

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
        error("solver #1 failed to solve boundary dofs (you should not see this message).")
    end

    # solve interior domain using LDLt factorization
    if F == nothing
        F = ldltfact(K[I,I])
    end
    u[I] = F \ (f[I] - K[I,B]*u[B])
    # solve lambda
    la[B] = lufact(C1[B,nz]) \ full(f[B] - K[B,I]*u[I] - K[B,B]*u[B])

    return F, true
end

"""
Solve linear system using LU factorization (UMFPACK). This version solves
directly the saddle point problem without elimination of boundary conditions.
"""
function solve!(K, C1, C2, D, f, g, u, la, ::Type{Val{2}}; F=nothing)
    # construct global system Ax = b and solve using lufact (UMFPACK)
    A = [K C1'; C2  D]
    b = [f; g]
    nz = get_nonzero_rows(A)
    x = zeros(length(b))
    if F == nothing
        F = lufact(A[nz,nz])
    end
    x[nz] = F \ full(b[nz])
    ndofs = size(K, 1)
    u[:] = x[1:ndofs]
    la[:] = x[ndofs+1:end]
    return F, true
end

""" Default linear system solver for solver. """
function solve_linear_system(solver::Solver; F=nothing, empty_assemblies_before_solution=true, show_info=true)
    show_info && info("Solving problems ...")
    t0 = Base.time()

    # assemble field & boundary problems
    # TODO: return same kind of set for both assembly types
    # M1, K1, Kg1, f1, fg1, C11, C21, D1, g1 = get_field_assembly(solver)
    # M2, K2, Kg2, f2, fg2, C12, C22, D2, g2 = get_boundary_assembly(solver)

    M, K, Kg, f, fg = get_field_assembly(solver)
    Kb, C1, C2, D, fb, g = get_boundary_assembly(solver)
    K = K + Kg + Kb
    f = f + fg + fb
    K = 1/2*(K + K')
    M = 1/2*(M + M')

    # free up some memory before solution
    for problem in get_problems(solver)
        if empty_assemblies_before_solution
            empty!(problem.assembly)
        else
            optimize!(problem.assembly)
        end
        gc()
    end

    u = zeros(solver.ndofs)
    la = zeros(solver.ndofs)

    status = false
    i = 0
    for i in [1, 2]
        F, status = solve!(K, C1, C2, D, f, g, u, la, Val{i}; F=F)
        status && break
    end
    status || error("Failed to solve linear system!")

    t1 = round(Base.time()-t0, 2)
    norms = (norm(u), norm(la))
    show_info && info("Solved problems in $t1 seconds using solver $i. Solution norms = $norms.")
    push!(solver.norms, norms)
    return F, u, la
end

""" Default assembler for solver. """
function assemble!(solver::Solver; show_info=true)
    show_info && info("Assembling problems ...")
    t0 = Base.time()
    nproblems = 0
    ndofs = 0
    for problem in solver.problems
        empty!(problem.assembly)
        assemble!(problem, solver.time)
        nproblems += 1
        ndofs = max(ndofs, size(problem.assembly.K, 2))
    end
    solver.ndofs = ndofs
    t1 = round(Base.time()-t0, 2)
    show_info && info("Assembled $nproblems problems in $t1 seconds. ndofs = $ndofs.")
end

""" Default initializer for solver. """
function initialize!(solver::Solver; show_info=true)
    show_info && info("Initializing problems ...")
    problems = get_problems(solver)
    length(problems) != 0 || error("Empty solver, add problems to solver using push!")
    t0 = Base.time()
    field_problems = get_field_problems(solver)
    length(field_problems) != 0 || warn("No field problem found from solver, add some..?")
    field_dim = get_unknown_field_dimension(first(field_problems))
    field_name = get_unknown_field_name(first(field_problems))
    info("initialize!(): looks we are solving $field_name, $field_dim dofs/node")
    nodes = Set{Int64}()
    for problem in problems
        initialize!(problem, solver.time)
        for element in get_elements(problem)
            conn = get_connectivity(element)
            push!(nodes, conn...)
        end
    end
    nnodes = length(nodes)
    info("Total number of nodes in problems: $nnodes")
    maxdof = maximum(nnodes)*field_dim
    info("# of max dof (=size of solution vector) is $maxdof")
    u = zeros(maxdof)
    la = zeros(maxdof)
    # TODO: this could be used to initialize elements too...
    for problem in problems
        problem.assembly.u = u
        problem.assembly.la = la
        # initialize(problem, ....)
    end
    t1 = round(Base.time()-t0, 2)
    show_info && info("Initialized problems in $t1 seconds.")
end

""" Default update for solver. """
function update!(solver::Solver, u::Vector, la::Vector; show_info=true)
    show_info && info("Updating problems ...")
    t0 = Base.time()
    for problem in solver.problems
        u_new, la_new = update_assembly!(problem, u, la)
        update_elements!(problem, u_new, la_new)
    end
    t1 = round(Base.time()-t0, 2)
    show_info && info("Updated problems in $t1 seconds.")
end

### Nonlinear quasistatic solver

type Nonlinear <: AbstractSolver
    iteration :: Int        # iteration counter
    min_iterations :: Int64 # minimum number of iterations
    max_iterations :: Int64 # maximum number of iterations
    convergence_tolerance :: Float64
    error_if_no_convergence :: Bool # throw error if no convergence
end

function Nonlinear()
    solver = Nonlinear(0, 1, 20, 5.0e-5, true)
    return solver
end

""" Check convergence of problems.

Notes
-----
Default convergence criteria is obtained by checking each sub-problem convergence.
"""
function has_converged(solver::Solver{Nonlinear}; show_info=false,
                       check_convergence_for_boundary_problems=false)
    properties = solver.properties
    converged = true
    eps = properties.convergence_tolerance
    for problem in solver.problems
        has_converged = true
        if is_field_problem(problem)
            has_converged = problem.assembly.u_norm_change < eps
            if isapprox(norm(problem.assembly.u), 0.0)
                # trivial solution
                has_converged = true
            end
            show_info && info("Details for problem $(problem.name)")
            show_info && info("Norm: $(norm(problem.assembly.u))")
            show_info && info("Norm change: $(problem.assembly.u_norm_change)")
            show_info && info("Has converged? $(has_converged)")
        end
        if is_boundary_problem(problem) && check_convergence_for_boundary_problems
            has_converged = problem.assembly.la_norm_change/norm(problem.assembly.la) < eps
            show_info && info("Details for problem $(problem.name)")
            show_info && info("Norm: $(norm(problem.assembly.la))")
            show_info && info("Norm change: $(problem.assembly.la_norm_change)")
            show_info && info("Has converged? $(has_converged)")
        end
        converged &= has_converged
    end
    return converged
end

type NonlinearConvergenceError <: Exception
    solver :: Solver
end

function Base.showerror(io::IO, exception::NonlinearConvergenceError)
    max_iters = exception.solver.properties.max_iterations
    print(io, "nonlinear iteration did not converge in $max_iters iterations!")
end

""" Default solver for quasistatic nonlinear problems. """
function call(solver::Solver{Nonlinear}; show_info=true)

    properties = solver.properties

    # 1. initialize each problem so that we can start nonlinear iterations
    initialize!(solver)

    # 2. start non-linear iterations
    for properties.iteration=1:properties.max_iterations
        show_info && info(repeat("-", 80))
        show_info && info("Starting nonlinear iteration #$(properties.iteration)")
        show_info && info("Increment time t=$(round(solver.time, 3))")
        show_info && info(repeat("-", 80))

        # 2.1 update linearized assemblies
        assemble!(solver)

        # 2.2 call solver for linearized system
        F, u, la = solve_linear_system(solver)

        # 2.3 update solution back to elements
        update!(solver, u, la)

        # 2.4 check convergence
        if has_converged(solver)
            info("Converged in $(properties.iteration) iterations.")
            properties.iteration >= properties.min_iterations && return true
            info("Convergence criteria met, but iteration < min_iterations, continuing...")
        end
    end

    # 3. did not converge
    properties.error_if_no_convergence && throw(NonlinearConvergenceError(solver))
end

""" Convenience function to call nonlinear solver. """
function NonlinearSolver(problems...)
    solver = Solver(Nonlinear, "default nonlinear solver")
    if length(problems) != 0
        push!(solver, problems...)
    end
    return solver
end
function NonlinearSolver(name::AbstractString, problems::Problem...)
    solver = NonlinearSolver(problems...)
    solver.name = name
    return solver
end


### Linear quasistatic solver

""" Quasistatic solver for linear problems.

Notes
-----
Main differences in this solver, compared to nonlinear solver are:
1. system of problems is assumed to converge in one step
2. reassembly of problem is done only if it's manually requested using empty!(problem.assembly)

"""
type Linear <: AbstractSolver
end

function assemble!(solver::Solver{Linear}; show_info=true)
    show_info && info("Assembling problems ...")
    tic()
    nproblems = 0
    ndofs = 0
    for problem in get_problems(solver)
        if isempty(problem.assembly)
            assemble!(problem, solver.time)
            nproblems += 1
        else
            show_info && info("$(problem.name) already assembled, skipping.")
        end
        ndofs = max(ndofs, size(problem.assembly.K, 2))
    end
    solver.ndofs = ndofs
    t1 = round(toq(), 2)
    show_info && info("Assembled $nproblems problems in $t1 seconds. ndofs = $ndofs.")
end

function call(solver::Solver{Linear}; F=nothing, show_info=true, return_factorization=true)
    t0 = Base.time()
    show_info && info(repeat("-", 80))
    show_info && info("Starting linear solver")
    show_info && info("Increment time t=$(round(solver.time, 3))")
    show_info && info(repeat("-", 80))
    initialize!(solver)
    assemble!(solver)
    F, u, la = solve_linear_system(solver; F=F, empty_assemblies_before_solution=false)
    update!(solver, u, la)
    t1 = round(Base.time()-t0, 2)
    show_info && info("Linear solver ready in $t1 seconds.")
    if return_factorization
        return F
    end
end

""" Convenience function to call linear solver. """
function LinearSolver(problems::Problem...)
    solver = Solver(Linear, "default linear solver")
    if length(problems) != 0
        push!(solver, problems...)
    end
    return solver
end
function LinearSolver(name::AbstractString, problems::Problem...)
    solver = LinearSolver(problems...)
    solver.name = name
    return solver
end

### End of linear quasistatic solver

### Postprocessor

type Postprocessor <: AbstractSolver
    assembly :: Assembly
    F :: Union{Factorization, Void}
end

function Postprocessor()
    Postprocessor(Assembly(), nothing)
end

function assemble!(solver::Solver{Postprocessor}; show_info=true)
    show_info && info("Assembling problems ...")
    tic()
    nproblems = 0
    ndofs = 0
    assembly = solver.properties.assembly
    empty!(assembly)
    for problem in get_problems(solver)
        for element in get_elements(problem)
            postprocess!(assembly, problem, element, solver.time)
        end
        nproblems += 1
        ndofs = max(ndofs, size(problem.assembly.K, 2))
    end
    solver.ndofs = ndofs
    t1 = round(toq(), 2)
    show_info && info("Assembled $nproblems problems in $t1 seconds. ndofs = $ndofs.")
end

function call(solver::Solver{Postprocessor}; show_info=true)
    t0 = Base.time()
    show_info && info(repeat("-", 80))
    show_info && info("Starting postprocessor")
    show_info && info("Increment time t=$(round(solver.time, 3))")
    show_info && info(repeat("-", 80))
    initialize!(solver)
    assemble!(solver)
    assembly = solver.properties.assembly
    M = sparse(assembly.M)
    f = sparse(assembly.f)
    F = cholfact(M)
    q = F \ f
    t1 = round(Base.time()-t0, 2)
    show_info && info("Postprocess of results ready in $t1 seconds.")
    return q
end

""" Convenience function to call postprocessor. """
function Postprocessor(problems::Problem...)
    solver = Solver(Postprocessor, "default postprocessor")
    if length(problems) != 0
        push!(solver, problems...)
    end
    return solver
end

function Postprocessor(name::AbstractString, problems::Problem...)
    solver = Postprocessor(problems...)
    solver.name = name
    return solver
end

