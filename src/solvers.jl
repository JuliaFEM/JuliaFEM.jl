# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

abstract AbstractSolver

type LinearSystem{Tv, Ti<:Integer}
    K :: SparseMatrixCSC{Tv, Ti}
    C1 :: SparseMatrixCSC{Tv, Ti}
    C2 :: SparseMatrixCSC{Tv, Ti}
    D :: SparseMatrixCSC{Tv, Ti}
    f :: SparseVector{Tv, Ti}
    g :: SparseVector{Tv, Ti}
    du :: SparseVector{Tv, Ti}
    la :: SparseVector{Tv, Ti}
end

function LinearSystem()
    K = SparseMatrixCSC(Matrix{Float64}())
    C1 = SparseMatrixCSC(Matrix{Float64}())
    C2 = SparseMatrixCSC(Matrix{Float64}())
    D = SparseMatrixCSC(Matrix{Float64}())
    f = SparseVector(Vector{Float64}())
    g = SparseVector(Vector{Float64}())
    du = SparseVector(Vector{Float64}())
    la = SparseVector(Vector{Float64}())
    return LinearSystem(K, C1, C2, D, f, g, du, la)
end

type Solver{S<:AbstractSolver}
    name :: String       # some descriptive name for problem
    time :: Float64              # current time
    problems :: Vector{Problem}
    norms :: Vector{Tuple}       # solution norms for convergence studies
    nnodes :: Int64              # total number of nodes in problem
    ndim :: Int64                # total number of dofs / node
    ndofs :: Int                 # total number of degrees of freedom in problems, nnodes*ndim
    dof_names :: Vector{String}  # name of dofs, ["U1", "U2", "U3", ...]
    xdmf :: Nullable{Xdmf}       # input/output handle
    initialized :: Bool
    ls :: LinearSystem{Float64, Int64}
    ls_prev :: LinearSystem{Float64, Int64}
    u :: SparseVector{Float64, Int64}
    la :: SparseVector{Float64, Int64}
    alpha :: Float64             # generalized alpha time integration coefficient
    fields :: Dict{String, Field}
    properties :: S
end

function Solver{S<:AbstractSolver}(::Type{S}, problems::Problem...)
    name = "$(S)Solver"
    time = 0.0
    problems_ = Problem[problem for problem in problems]
    norms = Tuple[]
    nnodes = 0
    ndim = 0
    ndofs = 0
    dof_names = String[]
    xdmf = nothing
    initialized = false
    ls = LinearSystem()
    ls_prev = LinearSystem()
    u = spzeros(ndofs)
    la = spzeros(ndofs)
    alpha = 0.0
    fields = Dict{String, Field}()
    properties = S()

    solver = Solver{S}(name, time, problems_, norms, nnodes, ndim, ndofs, dof_names, xdmf,
                       initialized, ls, ls_prev, u, la, alpha, fields, properties)

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

# one-liner helpers to identify problem types

is_field_problem(problem) = false
is_field_problem{P<:FieldProblem}(problem::Problem{P}) = true
is_boundary_problem(problem) = false
is_boundary_problem{P<:BoundaryProblem}(problem::Problem{P}) = true
get_field_problems(solver::Solver) = filter(is_field_problem, get_problems(solver))
get_boundary_problems(solver::Solver) = filter(is_boundary_problem, get_problems(solver))

"""Return one combined assembly for a set of field problems. """
function get_assembly(solver::Solver)

    M = SparseMatrixFEM()
    K = SparseMatrixFEM()
    Kg = SparseMatrixFEM()
    f = SparseVectorFEM()
    fg = SparseVectorFEM()
    C1 = SparseMatrixFEM()
    C2 = SparseMatrixFEM()
    D = SparseMatrixFEM()
    g = SparseVectorFEM()

    problems = get_field_problems(solver)

    for problem in problems
        add!(M, problem.assembly.M)
        add!(K, problem.assembly.K)
        add!(Kg, problem.assembly.Kg)
        add!(f, problem.assembly.f)
        add!(fg, problem.assembly.fg)
        add!(C1, problem.assembly.C1)
        add!(C2, problem.assembly.C2)
        add!(D, problem.assembly.D)
        add!(g, problem.assembly.g)
    end

    problems = get_boundary_problems(solver)

    already_constrained_dofs = Int64[]

    for problem in problems

        # if we are ordered to remove some rows (=dofs) from constraint matrices
        # before combining, do it now
        for dof in problem.assembly.removed_dofs
            info("$(problem.name): removing dof $dof from assembly")
            remove_row!(problem.assembly.C1, dof)
            remove_row!(problem.assembly.C2, dof)
            remove_row!(problem.assembly.D, dof)
        end

        # do a simple check that we do not have overconstrained dofs in system
        # this is just for case, we have tested this already
        nz_C2 = FEMSparse.get_nonzero_rows(problem.assembly.C2)
        nz_D = FEMSparse.get_nonzero_rows(problem.assembly.D)
        new_constraints = union(nz_C2, nz_D)
        overconstrained_dofs = intersect(already_constrained_dofs, new_constraints)
        if length(overconstrained_dofs) != 0
            sort!(overconstrained_dofs)
            sort!(already_constrained_dofs)
            sort!(new_constraints)
            warn("overconstrained dofs $overconstrained_dofs")
            warn("already constrained = $already_constrained_dofs")
            warn("new constraints = $new_constraints")
            warn("problem is overconstrained, finding overconstrained dofs... ")
            for dof in overconstrained_dofs
                for problem_ in problems
                    nz_C2 = get_nonzero_rows(problem_.assembly.C2)
                    nz_D = get_nonzero_rows(problem_.assembly.D)
                    new_constraints_ = union(nz_C2, nz_D)
                    new_constraints_ = setdiff(new_constraints_, problem_.assembly.removed_dofs)
                    if dof in new_constraints_
                        warn("overconstrained dof $dof defined in problem $(problem_.name)")
                    end
                end
                warn("To solve overconstrained situation, remove dofs from problems so that it exists only in one.")
                warn("To do this, use push! to add dofs to remove to problem.assembly.removed_dofs, e.g.")
                warn("`push!(bc.assembly.removed_dofs, $dof`)")
            end
            error("overconstrained dofs, not solving problem.")
        end
        already_constrained_dofs = union(already_constrained_dofs, new_constraints)

        add!(M, problem.assembly.M)
        add!(K, problem.assembly.K)
        add!(Kg, problem.assembly.Kg)
        add!(f, problem.assembly.f)
        add!(fg, problem.assembly.fg)
        add!(C1, problem.assembly.C1)
        add!(C2, problem.assembly.C2)
        add!(D, problem.assembly.D)
        add!(g, problem.assembly.g)
    end

    if length(K) == 0
        warn("get_assembly(): stiffness matrix seems to be empty.")
        warn("get_assembly(): check that elements are pushed to problem and formulation is correct.")
    end

    if length(C2) == 0
        warn("get_boundary_assembly(): constraint matrix seems to be empty.")
        warn("get_boundary_assembly(): check that boundary elements are pushed to problem and formulation is correct.")
    end

    return M, K, Kg, f, fg, C1, C2, D, g

end

include("solvers_linear_system_solver.jl")

""" Default linear system solver for solver. """
function solve!(solver::Solver; empty_field_assemblies_before_solution=true, symmetric=true)

    info("Solving problems ...")
    t0 = Base.time()

    # assemble field & boundary problems
    @timeit to "combine assemblies" M, K, Kg, f, fg, C1, C2, D, g = get_assembly(solver)

    if empty_field_assemblies_before_solution
        @timeit to "empty field assemblies before solution" begin
            # free up some memory before solution by emptying field assemblies from problems
            for problem in get_field_problems(solver)
                empty!(problem.assembly)
            end
            gc()
        end
    end

    # assembling of problems is almost ready, convert to CSC format for fast arithmetic operations

    if solver.ndofs == 0
        K_size = size(K)
        @assert K_size[1] == K_size[2]
        solver.ndofs = K_size[1]
        info("solve!(): automatically determined problem dimension, ndofs = $(solver.ndofs)")
    end

    ndofs = solver.ndofs
    length(solver.ls.la) == 0 && (solver.ls.la = spzeros(ndofs))
    length(solver.ls.du) == 0 && (solver.ls.du = spzeros(ndofs))
    length(solver.u) == 0 && (solver.u = spzeros(ndofs))
    length(solver.la) == 0 && (solver.la = spzeros(ndofs))

    @timeit to "construct LinearSystem" begin
        ls = solver.ls
        ls_prev = solver.ls_prev
        ls.K = sparse(K, ndofs, ndofs)# + sparse(Kg, ndofs, ndofs)
        ls.C1 = sparse(C1, ndofs, ndofs)
        ls.C2 = sparse(C2, ndofs, ndofs)
        ls.D = sparse(D, ndofs, ndofs)
        ls.f = sparsevec(f, ndofs)# + sparsevec(f, ndofs)
        ls.g = sparsevec(g, ndofs)
    end

    if symmetric
        @timeit to "make stffiness matrix symmetric" ls.K = 1/2*(ls.K + ls.K')
    end

    # kick in generalized alpha rule for time integration
    if !isapprox(solver.alpha, 0.0) && !isempty(ls_prev.f)
        @timeit to "use generalized-alpha time integration" begin
            alpha = solver.alpha
            debug("Using generalized-α time integration, α=$alpha")
            ls.K = (1-alpha)*ls.K
            ls.C1 = (1-alpha)*ls.C1
            ls.f = (1-alpha)*ls.f + alpha*ls_prev.f
        end
    end

    is_solved = false
    i = 0
    @timeit to "solve linear system" for i in [1, 2, 3]
        is_solved = solve_linear_system!(ls, Val{i})
        if is_solved
            break
        end
    end
    if !is_solved
        error("Failed to solve linear system!")
    end
    t1 = round(Base.time()-t0, 2)

    norms = (norm(ls.du), norm(ls.la))
    push!(solver.norms, norms)

    info("Solved problems in $t1 seconds using solver $i.")
    info("Solution norms = $norms.")

    @timeit to "update solution vectors" begin
        ls_prev.f = ls.f
        solver.u += ls.du
        solver.la = ls.la
    end

    return nothing
end

""" Default assembler for solver. """
function assemble!(solver::Solver; timing=true, with_mass_matrix=false)
    info("Assembling problems ...")

    function do_assemble(problem)
        t00 = Base.time()
        empty!(problem.assembly)
        assemble!(problem, solver.time)
        if with_mass_matrix && is_field_problem(problem)
            assemble!(problem, solver.time, Val{:mass_matrix})
        end
        t11 = Base.time()
        return t11-t00
    end

    t0 = Base.time()
    assembly_times = map(do_assemble, solver.problems)
    nproblems = length(assembly_times)

    ndofs = 0
    for problem in solver.problems
        Ks = size(problem.assembly.K, 2)
        Cs = size(problem.assembly.C1, 2)
        ndofs = max(ndofs, Ks, Cs)
    end

    solver.ndofs = ndofs
    t1 = round(Base.time()-t0, 2)
    info("Assembled $nproblems problems in $t1 seconds. ndofs = $ndofs.")
    if timing
        info("Assembly times:")
        for (i, problem) in enumerate(solver.problems)
            pn = problem.name
            pt = round(assembly_times[i], 2)
            info("$i $pn $pt")
        end
    end
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
        return nothing
    end
    for problem in get_problems(solver)
        initialize!(problem, solver.time)
    end
    solver.initialized = true
    return nothing
end

function to_dict{Tv,Ti<:Integer}(b::SparseVector{Tv,Ti}, dim::Int)
    I, V = findnz(b)
    debug(I)
    debug(V)
    DOK = Dict{Ti,Tv}(i => v for (i, v) in zip(I, V))
    dim == 1 && return DOK
    node_ids = unique(ceil(Int, I/dim))
    nnodes = length(node_ids)
    debug("number of nodes = $nnodes")
    debug("node ids = $node_ids")
    MDOK = Dict{Ti,Vector{Tv}}()
    for nid in node_ids
        val = [get(DOK, dim*(nid-1)+i, Tv(0)) for i=1:dim]
        debug("$nid => $val")
        MDOK[nid] = val
    end
    return MDOK
end

""" Default update for solver. """
function update!{S}(solver::Solver{S})
    @assert length(get_unknown_fields(solver)) == 1
    dim = solver.ndim = get_unknown_field_dimension(solver)
    time = solver.time
    info("Updating problems, $dim dofs / node ...")

    SparseArrays.droptol!(solver.u, 1.0e-12)
    SparseArrays.droptol!(solver.la, 1.0e-12)

    u = to_dict(solver.u, dim)
    la = to_dict(solver.la, dim)

    node_ids = Set{Int64}()
    for problem in get_problems(solver)
        for element in get_elements(problem)
            for c in get_connectivity(element)
                push!(node_ids, c)
            end
        end
    end

    solver.nnodes = length(node_ids)

    for nid in node_ids
        if !haskey(u, nid)
            u[nid] = dim == 1 ? 0.0 : zeros(dim)
        end
        if !haskey(la, nid)
            la[nid] = dim == 1 ? 0.0 : zeros(dim)
        end
    end

    t0 = Base.time()

    for problem in get_field_problems(solver)
        field_name = get_unknown_field_name(problem)
        for element in get_elements(problem)
            connectivity = get_connectivity(element)
            update!(element, field_name, time => [u[i] for i in connectivity])
        end
    end

    for problem in get_boundary_problems(solver)
        field_name = get_unknown_field_name(problem) # "lambda"
        parent_field_name = get_parent_field_name(problem)
        for element in get_elements(problem)
            connectivity = get_connectivity(element)
            update!(element, parent_field_name, time => [u[i] for i in connectivity])
            update!(element, field_name, time => [la[i] for i in connectivity])
        end
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
    @timeit to "initialize!(solver)" initialize!(solver)

    # 2. start non-linear iterations
    for properties.iteration=1:properties.max_iterations
        info(repeat("-", 80))
        info("Starting nonlinear iteration #$(properties.iteration)")
        info("Increment time t=$(round(solver.time, 3))")
        info(repeat("-", 80))

        # 2.1 update assemblies
        @timeit to "assemble!(solver)" assemble!(solver)

        # 2.2 call solver for linearized system
        @timeit to "solve!(solver)" solve!(solver)

        # 2.3 update solution back to elements
        @timeit to "update!(solver)" update!(solver)

        # 2.4 check convergence
        if properties.iteration >= properties.min_iterations && has_converged(solver)
            info("Converged in $(properties.iteration) iterations.")
            # 2.4.1 run any postprocessing of problems
            @timeit to "postprocess!(solver)" postprocess!(solver)
            # 2.4.2 update Xdmf output
            @timeit to "update_xdmf!(solver)" update_xdmf!(solver)
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
function NonlinearSolver(name::String, problems::Problem...)
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
    @timeit to "initialize!(solver)" initialize!(solver)
    @timeit to "assemble!(solver)" assemble!(solver)
    @timeit to "solve!(solver)" solve!(solver)
    @timeit to "update!(solver)" update!(solver)
    t1 = round(Base.time()-t0, 2)
    info("Linear solver ready in $t1 seconds.")
end

""" Convenience function to call linear solver. """
function LinearSolver(problems::Problem...)
    solver = Solver(Linear, problems...)
    return solver
end
function LinearSolver(name::String, problems::Problem...)
    solver = Solver(Linear, problems...)
    solver.name = name
    return solver
end

### End of linear quasistatic solver
