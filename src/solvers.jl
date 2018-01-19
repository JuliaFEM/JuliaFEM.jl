# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

abstract type AbstractSolver end

type Solver{S<:AbstractSolver}
    name :: AbstractString       # some descriptive name for problem
    time :: Float64              # current time
    problems :: Vector{Problem}
    norms :: Vector{Tuple}       # solution norms for convergence studies
    ndofs :: Int                 # number of degrees of freedom in problem
    xdmf :: Nullable{Xdmf}       # input/output handle
    initialized :: Bool
    u :: Vector{Float64}
    la :: Vector{Float64}
    alpha :: Float64             # generalized alpha time integration coefficient
    fields :: Dict{String, AbstractField}
    properties :: S
end


function Solver{S<:AbstractSolver}(::Type{S}, name="solver", properties...)
    variant = S(properties...)
    solver = Solver{S}(name, 0.0, [], [], 0, nothing, false, [], [], 0.0, Dict(), variant)
    return solver
end

function Solver{S<:AbstractSolver}(::Type{S}, problems::Problem...)
    solver = Solver(S, "$(S)Solver")
    push!(solver.problems, problems...)
    return solver
end

function get_problems(solver::Solver)
    return solver.problems
end

function push!(solver::Solver, problem::Problem)
    push!(solver.problems, problem)
end

function getindex(solver::Solver, problem_name::String)
    for problem in get_problems(solver)
        if problem.name == problem_name
            return problem
        end
    end
    throw(KeyError(problem_name))
end

function haskey(solver::Solver, field_name::String)
    return haskey(solver.fields, field_name)
end

get_field_problems(solver::Solver) = filter(is_field_problem, get_problems(solver))
get_boundary_problems(solver::Solver) = filter(is_boundary_problem, get_problems(solver))

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
function get_field_assembly(solver::Solver)
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
        info("automatically determined problem dimension, ndofs = $(solver.ndofs)")
    end

    M = sparse(M, solver.ndofs, solver.ndofs)
    K = sparse(K, solver.ndofs, solver.ndofs)
    if nnz(K) == 0
        warn("Field assembly seems to be empty. Check that elements are pushed to problem and formulation is correct.")
    end
    Kg = sparse(Kg, solver.ndofs, solver.ndofs)
    f = sparse(f, solver.ndofs, 1)
    fg = sparse(fg, solver.ndofs, 1)

    return M, K, Kg, f, fg
end

""" Loop through boundary assemblies and check for possible overconstrain situations. """
function check_for_overconstrained_dofs(solver::Solver)
    overdetermined = false
    constrained_dofs = Set{Int}()
    all_overconstrained_dofs = Set{Int}()
    boundary_problems = get_boundary_problems(solver)
    for problem in boundary_problems
        new_constraints = Set(problem.assembly.C2.I)
        new_constraints = setdiff(new_constraints, problem.assembly.removed_dofs)
        overconstrained_dofs = intersect(constrained_dofs, new_constraints)
        all_overconstrained_dofs = union(all_overconstrained_dofs, overconstrained_dofs)
        if length(overconstrained_dofs) != 0
            warn("problem is overconstrained, finding overconstrained dofs... ")
            overdetermined = true
            for dof in overconstrained_dofs
                for problem_ in boundary_problems
                    new_constraints_ = Set(problem_.assembly.C2.I)
                    new_constraints_ = setdiff(new_constraints_, problem_.assembly.removed_dofs)
                    if dof in new_constraints_
                        warn("overconstrained dof $dof defined in problem $(problem_.name)")
                    end
                end
                warn("To solve overconstrained situation, remove dofs from problems so that it exists only in one.")
                warn("To do this, use push! to add dofs to remove to problem.assembly.removed_dofs, e.g.")
                warn("`push!(bc.assembly.removed_dofs, $dof`)")
            end
        end
        constrained_dofs = union(constrained_dofs, new_constraints)
    end
    if overdetermined
        warn("List of all overconstrained dofs:")
        warn(sort(collect(all_overconstrained_dofs)))
        error("problem is overconstrained, not continuing to solution.")
    end
    return true
end

""" Return one combined boundary assembly for a set of boundary problems.

Returns
-------
K, C1, C2, D, f, g :: SparseMatrixCSC

"""
function get_boundary_assembly(solver::Solver)

    check_for_overconstrained_dofs(solver)

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
        for dof in assembly.removed_dofs
            info("$(problem.name): removing dof $dof from assembly")
            C1_[dof,:] = 0.0
            C2_[dof,:] = 0.0
        end
        SparseArrays.dropzeros!(C1_)
        SparseArrays.dropzeros!(C2_)

        already_constrained = get_nonzero_rows(C2)
        new_constraints = get_nonzero_rows(C2_)
        overconstrained_dofs = intersect(already_constrained, new_constraints)
        if length(overconstrained_dofs) != 0
            warn("overconstrained dofs $overconstrained_dofs")
            warn("already constrained = $already_constrained")
            warn("new constraints = $new_constraints")
            overconstrained_dofs = sort(overconstrained_dofs)
            error("overconstrained dofs, not solving problem.")
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


"""
Solve linear system using LDLt factorization (SuiteSparse). This version
requires that final system is symmetric and positive definite, so boundary
conditions are first eliminated before solution.
"""
function solve!(solver::Solver, K, C1, C2, D, f, g, u, la, ::Type{Val{1}})

    nnz(D) == 0 || return false

    A = get_nonzero_rows(K)
    B = get_nonzero_rows(C2)
    B2 = get_nonzero_columns(C2)
    B == B2 || return false
    I = setdiff(A, B)
    
    debug("# A = $(length(A))")
    debug("# B = $(length(B))")
    debug("# I = $(length(I))")

    if length(B) == 0
        warn("No rows in C2, forget to set Dirichlet boundary conditions to model?")
    else
        u[B] = lufact(C2[B,B2]) \ full(g[B])
    end

    # solve interior domain using LDLt factorization
    F = ldltfact(K[I,I])
    u[I] = F \ (f[I] - K[I,B]*u[B])

    # solve lagrange multipliers
    la[B] = lufact(C1[B2,B]) \ full(f[B] - K[B,I]*u[I] - K[B,B]*u[B])

    return true
end

"""
Solve linear system using LU factorization (UMFPACK). This version solves
directly the saddle point problem without elimination of boundary conditions.
It is assumed that C1 == C2 and D = 0, so problem is symmetric and zero rows
cand be removed from total system before solution. This kind of system arises
in e.g. mesh tie problem
"""
function solve!(solver::Solver, K, C1, C2, D, f, g, u, la, ::Type{Val{2}})

    C1 == C2 || return false
    length(D) == 0 || return false

    A = [K C1'; C2  D]
    b = [f; g]

    nz1 = get_nonzero_rows(A)
    nz2 = get_nonzero_columns(A)
    nz1 == nz2 || return false
    
    x = zeros(2*solver.ndofs)
    x[nz1] = lufact(A[nz1,nz2]) \ full(b[nz1])

    u[:] = x[1:solver.ndofs]
    la[:] = x[solver.ndofs+1:end]

    return true
end

"""
Solve linear system using LU factorization (UMFPACK). This version solves
directly the saddle point problem without elimination of boundary conditions.
If matrix has zero rows, diagonal term is added to that matrix is invertible.
"""
function solve!(solver::Solver, K, C1, C2, D, f, g, u, la, ::Type{Val{3}})

    A = [K C1'; C2  D]
    b = [f; g]

    nz = ones(2*solver.ndofs)
    nz[get_nonzero_rows(A)] = 0.0
    A += spdiagm(nz)

    x = lufact(A) \ full(b)

    u[:] = x[1:solver.ndofs]
    la[:] = x[solver.ndofs+1:end]

    return true
end

""" Default linear system solver for solver. """
function solve!(solver::Solver; empty_assemblies_before_solution=true, symmetric=true)

    info("Solving problems ...")
    t0 = Base.time()

    # assemble field & boundary problems
    # TODO: return same kind of set for both assembly types
    # M1, K1, Kg1, f1, fg1, C11, C21, D1, g1 = get_field_assembly(solver)
    # M2, K2, Kg2, f2, fg2, C12, C22, D2, g2 = get_boundary_assembly(solver)

    M, K, Kg, f, fg = get_field_assembly(solver)
    Kb, C1, C2, D, fb, g = get_boundary_assembly(solver)
    K = K + Kg + Kb
    f = f + fg + fb

    if symmetric
        K = 1/2*(K + K')
        M = 1/2*(M + M')
    end

    if empty_assemblies_before_solution
        # free up some memory before solution by emptying field assemblies from problems
        for problem in get_field_problems(solver)
            empty!(problem.assembly)
        end
        gc()
    end
    
    if !haskey(solver, "fint")
        solver.fields["fint"] = field(solver.time => f)
    else
        update!(solver.fields["fint"], solver.time => f)
    end

    fint = solver.fields["fint"]

    if length(fint) > 1
        # kick in generalized alpha rule for time integration
        alpha = solver.alpha
        debug("Using generalized-α time integration, α=$alpha")
        K = (1-alpha)*K
        C1 = (1-alpha)*C1
        f = (1-alpha)*f + alpha*fint.data[end-1].second
    end

    ndofs = solver.ndofs
    u = zeros(ndofs)
    la = zeros(ndofs)
    is_solved = false
    i = 0
    for i in [1, 2, 3]
        is_solved = solve!(solver, K, C1, C2, D, f, g, u, la, Val{i})
        if is_solved
            break
        end
    end
    if !is_solved
        error("Failed to solve linear system!")
    end
    t1 = round(Base.time()-t0, 2)
    norms = (norm(u), norm(la))
    push!(solver.norms, norms)

    solver.u = u
    solver.la = la

    info("Solved problems in $t1 seconds using solver $i.")
    info("Solution norms = $norms.")

    return
end

"""
    assemble!(solver; with_mass_matrix=false)

Default assembler for solver.

This function loops over all problems defined in problem and launches
standard assembler for them. As a result, each problem.assembly is
populated with global stiffness matrix, force vector, and, optionally,
mass matrix.
"""
function assemble!(solver::Solver; with_mass_matrix=false)
    info("Assembling problems ...")

    for problem in get_problems(solver)
        timeit("assemble $(problem.name)") do
            empty!(problem.assembly)
            assemble!(problem, solver.time)
        end
    end

    if with_mass_matrix
        for problem in get_field_problems(solver)
            timeit("assemble $(problem.name) mass matrix") do
                assemble!(problem, solver.time, Val{:mass_matrix})
            end
        end
    end

    ndofs = 0
    for problem in solver.problems
        Ks = size(problem.assembly.K, 2)
        Cs = size(problem.assembly.C1, 2)
        ndofs = max(ndofs, Ks, Cs)
    end
    solver.ndofs = ndofs

    info("Assembly done!")
end

function get_unknown_fields(solver::Solver)
    fields = Dict()
    for problem in get_field_problems(solver)
        field_name = get_unknown_field_name(problem)
        field_dim = get_unknown_field_dimension(problem)
        fields[field_name] = field_dim
    end
    return fields
end

function get_unknown_field_name(solver::Solver)
    fields = get_unknown_fields(solver)
    return join(sort(collect(keys(fields))), ", ")
end

function get_unknown_field_dimension(solver::Solver)
    fields = get_unknown_fields(solver)
    return sum(values(fields))
end

""" Default initializer for solver. """
function initialize!(solver::Solver)
    if solver.initialized
        warn("initialize!(): solver already initialized")
        return
    end
    info("Initializing solver ...")
    problems = get_problems(solver)
    length(problems) != 0 || error("Empty solver, add problems to solver using push!")
    t0 = Base.time()
    field_problems = get_field_problems(solver)
    length(field_problems) != 0 || warn("No field problem found from solver, add some..?")
    field_name = get_unknown_field_name(solver)
    field_dim = get_unknown_field_dimension(solver)
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
    maxdof = maximum(nodes)*field_dim
    info("# of max dof (=size of solution vector) is $maxdof")
    solver.u = zeros(maxdof)
    solver.la = zeros(maxdof)
    # TODO: this could be used to initialize elements too...
    # TODO: cannot initialize to zero always, construct vector from elements.
    for problem in problems
        problem.assembly.u = zeros(maxdof)
        problem.assembly.la = zeros(maxdof)
        # initialize(problem, ....)
    end
    t1 = round(Base.time()-t0, 2)
    info("Initialized solver in $t1 seconds.")
    solver.initialized = true
end

function get_all_elements(solver::Solver)
    elements = [get_elements(problem) for problem in get_problems(solver)]
    return [elements...;]
end

"""
Return nodal field from all problems defined in solver.

Examples
--------
To return e.g. geometry defined in nodal points at time t=0.0, one can write:

julia> solver("geometry", 0.0)
"""
function (solver::Solver)(field_name::String, time::Float64)
    fields = []
    for problem in get_problems(solver)
        field = problem(field_name, time)
        if field == nothing
            continue
        end
        if length(field) == 0
            warn("no field $field_name found for problem $(problem.name)")
            continue
        end
        push!(fields, field)
    end
    if length(fields) == 0
        return Dict{Integer, Vector{Float64}}()
    end
    return merge(fields...)
end

""" Default update for solver. """
function update!{S}(solver::Solver{S})
    u = solver.u
    la = solver.la

    info("Updating problems ...")
    t0 = Base.time()

    for problem in get_problems(solver)
        assembly = get_assembly(problem)
        elements = get_elements(problem)
        # update solution, first for assembly (u,la) ...
        update!(problem, assembly, u, la)
        # .. and then from assembly (u,la) to elements
        update!(problem, assembly, elements, solver.time)
    end

    t1 = round(Base.time()-t0, 2)
    info("Updated problems in $t1 seconds.")
end

""" Default postprocess for solver. Loop all problems and run postprocess
functions to calculate secondary fields, i.e. contact pressure, stress,
heat flux, reaction force etc. quantities.
"""
function postprocess!(solver::Solver)
    info("Running postprocess scripts for solver...")
    for problem in get_problems(solver)
        for field_name in problem.postprocess_fields
            field = Val{Symbol(field_name)}
            info("Running postprocess for problem $(problem.name), field $field_name")
            postprocess!(problem, solver.time, field)
        end
    end
end

""" Default xdmf update for solver. Loop all problems and write them individually
to Xdmf file. By default write the main unknown field (displacement, temperature,
...) and any fields requested separately in `problem.postprocess_fields` vector
(stress, strain, ...)
"""
function update_xdmf!(solver::Solver)
    if isnull(solver.xdmf)
        info("update_xdmf: xdmf not attached to solver, not writing output to file.")
        info("turn Xdmf writing on to solver by typing: solver.xdmf = Xdmf(\"results\")")
        return
    end
    xdmf = get(solver.xdmf)
    for problem in get_problems(solver)
        fields = [get_unknown_field_name(problem); problem.postprocess_fields]
        if is_boundary_problem(problem)
            fields = [fields; get_parent_field_name(problem)]
        end
        update_xdmf!(xdmf, problem, solver.time, fields)
    end
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
    solver = Nonlinear(0, 1, 10, 5.0e-5, true)
    return solver
end

""" Check convergence of problems.

Notes
-----
Default convergence criteria is obtained by checking each sub-problem convergence.
"""
function has_converged(solver::Solver{Nonlinear})
    properties = solver.properties
    converged = true
    eps = properties.convergence_tolerance
    for problem in get_field_problems(solver)
        has_converged = problem.assembly.u_norm_change < eps
        if isapprox(norm(problem.assembly.u), 0.0)
            # trivial solution
            has_converged = true
        end
        debug("Details for problem $(problem.name)")
        debug("Norm: $(norm(problem.assembly.u))")
        debug("Norm change: $(problem.assembly.u_norm_change)")
        debug("Has converged? $(has_converged)")
        converged &= has_converged
    end
    return converged
end

""" Default solver for quasistatic nonlinear problems. """
function (solver::Solver{Nonlinear})()

    properties = solver.properties

    # 1. initialize each problem so that we can start nonlinear iterations
    initialize!(solver)

    # 2. start non-linear iterations
    for properties.iteration=1:properties.max_iterations
        info(repeat("-", 80))
        info("Starting nonlinear iteration #$(properties.iteration)")
        info("Increment time t=$(round(solver.time, 3))")
        info(repeat("-", 80))

        # 2.1 update assemblies
        assemble!(solver)

        # 2.2 call solver for linearized system
        solve!(solver)

        # 2.3 update solution back to elements
        update!(solver)

        # 2.4 check convergence
        if properties.iteration >= properties.min_iterations && has_converged(solver)
            info("Converged in $(properties.iteration) iterations.")
            # 2.4.1 run any postprocessing of problems
            postprocess!(solver)
            # 2.4.2 update Xdmf output
            update_xdmf!(solver)
            return true
        end
    end

    # 3. did not converge
    if properties.error_if_no_convergence
        error("nonlinear iteration did not converge in $(properties.iteration) iterations!")
    end
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

function assemble!(solver::Solver{Linear})
    info("Assembling problems ...")
    tic()
    nproblems = 0
    ndofs = 0
    for problem in get_problems(solver)
        if isempty(problem.assembly)
            assemble!(problem, solver.time)
            nproblems += 1
        else
            info("$(problem.name) already assembled, skipping.")
        end
        ndofs = max(ndofs, size(problem.assembly.K, 2))
    end
    solver.ndofs = ndofs
    t1 = round(toq(), 2)
    info("Assembled $nproblems problems in $t1 seconds. ndofs = $ndofs.")
end

function (solver::Solver{Linear})()
    t0 = Base.time()
    info(repeat("-", 80))
    info("Starting linear solver")
    info("Increment time t=$(round(solver.time, 3))")
    info(repeat("-", 80))
    @timeit "initialize solver" initialize!(solver)
    @timeit "assemble problems" assemble!(solver)
    @timeit "solve linear system" solve!(solver)
    @timeit "update problems" update!(solver)
    t1 = round(Base.time()-t0, 2)
    info("Linear solver ready in $t1 seconds.")
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
