# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

## Direct solver

using JuliaFEM

type DirectSolver
    name :: ASCIIString
    field_problems :: Vector{Problem}
    boundary_problems :: Vector{BoundaryProblem}
    parallel :: Bool
    solve_residual :: Bool
    nonlinear_max_iterations :: Int64
    nonlinear_convergence_tolerance :: Float64
    linear_system_solver_preprocessors :: Vector{Tuple{Symbol,Any,Any}}
    linear_system_solvers :: Vector{Tuple{Symbol,Any,Any}}
    linear_system_solver_postprocessors :: Vector{Tuple{Symbol,Any,Any}}
end

""" Default initializer. """
function DirectSolver(name="DirectSolver")
    DirectSolver(
        name,
        [],                             # field problems
        [],                             # boundary problems
        false,                          # parallel run?
        true,                           # solve residual or total quantity
        10,                             # nonlinear problem max iterations
        5.0e-6,                         # nonlinear convergence tolerance
        [],                             # default solution preprocessors
        [(:UMFPACK, (), [])],           # linear system solver: CHOLMOD, UMFPACK
        [],                             # default solution postprocessors
    )
end

function set_name!(solver::DirectSolver, name::ASCIIString)
    solver.name = name
end

function set_linear_system_solver!(solver::DirectSolver, method::Symbol)
    solver.linear_system_solvers = [(method, (), [])]
end

function set_nonlinear_max_iterations!(solver::DirectSolver, max_iterations::Int)
    solver.nonlinear_max_iterations = max_iterations
end

function push!(solver::DirectSolver, problem::FieldProblem)
    push!(solver.field_problems, problem)
end

function push!(solver::DirectSolver, problem::BoundaryProblem)
    push!(solver.boundary_problems, problem)
end

function add_linear_system_solver_preprocessor!(solver::DirectSolver, preprocessor_name::Symbol, args...; kwargs...)
    push!(solver.linear_system_solver_preprocessors, (preprocessor_name, args, kwargs))
end

function add_linear_system_solver_postprocessor!(solver::DirectSolver, postprocessor_name::Symbol, args...; kwargs...)
    push!(solver.linear_system_solver_postprocessors, (postprocessor_name, args, kwargs))
end

function tic(timing, what::ASCIIString)
    timing[what * " start"] = time()
end

function toc(timing, what::ASCIIString)
    timing[what * " finish"] = time()
end

function time_elapsed(timing, what::ASCIIString)
    return timing[what * " finish"] - timing[what * " start"]
end

"""
Linear system solver for problem

    Ku  + C₁'λ = f
    C₂u + Dλ   = g

"""
function linear_system_solver_solve!(solver, iter, time, K, f, C1, C2, D, g, sol, la, ::Type{Val{:UMFPACK}})
    t0 = Base.time()
    dim = size(K, 1)
    A = [K C1'; C2 D]
    b = [f; g]
    nz1 = sort(unique(rowvals(A)))
    nz2 = sort(unique(rowvals(A')))
    u = zeros(length(b))
    u[nz1] = lufact(A[nz1,nz2]) \ full(b[nz1])
    sol[:] = u[1:dim]
    la[:] = u[dim+1:end]
    info("UMFPACK: solved in ", Base.time()-t0, " seconds. norm = ", norm(sol))
end

""" Solution preprocessor: dump matrices to disk before solution.

Examples
--------
julia> solver = DirectSolver()
julia> push!(solver.linear_system_solver_preprocessors, :dump_matrices)

"""
function linear_system_solver_preprocess!(solver, iter, time, K, f, C1, C2, D, g, sol, la, ::Type{Val{:dump_matrices}})
    filename = "matrices_$(solver.name)_host_$(myid())_iteration_$(iter).jld"
    info("dumping matrices to disk, file = $filename")
    save(filename,
        "stiffness matrix K", K,
        "force vector f", f,
        "constraint matrix C1", C1,
        "constraint matrix C2", C2,
        "constraint matrix D", D,
        "constraint vector g", g)
end

function linear_system_solver_postprocess!
end

""" Initialize unknown field ready for nonlinear iterations, i.e.,
    take last known value and set it as a initial quess for next
    time increment.
"""
function initialize!(problem::FieldProblem, time::Real)
    field_name = get_unknown_field_name(problem)
    field_dim = get_unknown_field_dimension(problem)
    for element in get_elements(problem)
        gdofs = get_gdofs(element, problem)
        if haskey(element, field_name)
            if !isapprox(last(element[field_name]).time, time)
                last_data = copy(last(element[field_name]).data)
                push!(element[field_name], time => last_data)
            end
        else # if field not found at all, initialize new zero field.
            data = Vector{Float64}[zeros(field_dim) for i in 1:length(element)]
            element[field_name] = (time => data)
        end
    end
end

function initialize!(problem::BoundaryProblem, time::Real; initialize_primary_field=false)
    field_name = problem.parent_field_name
    field_dim = problem.parent_field_dim

    for element in get_elements(problem)
        gdofs = get_gdofs(element, problem)
        data = Vector{Float64}[zeros(field_dim) for i in 1:length(element)]

        # add new field "reaction force" for boundary element if not found
        if haskey(element, "reaction force")
            if !isapprox(last(element["reaction force"]).time, time)
                push!(element["reaction force"], time => data)
            end
        else
            element["reaction force"] = (time => data)
        end

        if initialize_primary_field
            # add new primary field for boundary element if not found
            if haskey(element, field_name)
                if !isapprox(last(element[field_name]).time, time)
                    last_data = copy(last(element[field_name]).data)
                    push!(element[field_name], time => last_data)
                end
            else
                data = Vector{Float64}[zeros(field_dim) for i in 1:length(element)]
                element[field_name] = (time => data)
            end
        end
    end
end

function update!(problem::FieldProblem, solution::Vector, ::Type{Val{:elements}})
    field_name = get_unknown_field_name(problem)
    field_dim = get_unknown_field_dimension(problem)
    for element in get_elements(problem)
        gdofs = get_gdofs(element, problem)
        local_sol = solution[gdofs]
        local_sol = reshape(local_sol, field_dim, length(element))
        local_sol = Vector{Float64}[local_sol[:,i] for i=1:length(element)]
        last(element[field_name]).data = local_sol
    end
end

function update!(problem::BoundaryProblem, solution::Vector, ::Type{Val{:elements}})
    field_name = problem.parent_field_name
    field_dim = problem.parent_field_dim
    for element in get_elements(problem)
        gdofs = get_gdofs(element, field_dim)
        local_sol = solution[gdofs]
        local_sol = reshape(local_sol, field_dim, length(element))
        local_sol = Vector{Float64}[local_sol[:,i] for i=1:length(element)]
        last(element["reaction force"]).data = local_sol
    end
end


""" Call solver to solve a set of problems. """
function call(solver::DirectSolver, time::Real=0.0)
    info("Starting solver $(solver.name)")
    info("# of field problems: $(length(solver.field_problems))")
    info("# of boundary problems: $(length(solver.boundary_problems))")
    (length(solver.field_problems) != 0) || error("no field problems defined for solver, use push!(solver, problem, ...) to define field problems.")

    timing = Dict{ASCIIString, Float64}()
    tic(timing, "solver")

    # check that all problems are "same kind"
    field_name = get_unknown_field_name(solver.field_problems[1])
    field_dim = get_unknown_field_dimension(solver.field_problems[1])
    for field_problem in solver.field_problems
        get_unknown_field_name(field_problem) == field_name || error("several different fields not supported yet")
        get_unknown_field_dimension(field_problem) == field_dim || error("several different field dimensions not supported yet")
    end

    tic(timing, "initialization")
    for field_problem in solver.field_problems
        initialize!(field_problem, time)
    end
    for boundary_problem in solver.boundary_problems
        initialize!(boundary_problem, time)
    end
    toc(timing, "initialization")

    dim = nothing
    sol = nothing
    last_sol = nothing
    la = nothing
    last_la = nothing

    for iter=1:solver.nonlinear_max_iterations
        info("Starting nonlinear iteration $iter")
        tic(timing, "non-linear iteration")

        tic(timing, "field assembly")
        info("Assembling field problems...")
        field_assembly = FieldAssembly()
        for (i, problem) in enumerate(solver.field_problems)
            info("Assembling body $i: $(problem.name)")
            append!(field_assembly, assemble(problem, time))
        end
        K = sparse(field_assembly.stiffness_matrix)
        dim = size(K, 1)
        f = sparse(field_assembly.force_vector, dim, 1)
        field_assembly = nothing
        gc()
        toc(timing, "field assembly")

        tic(timing, "boundary assembly")
        info("Assembling boundary problems...")
        boundary_assembly = BoundaryAssembly()
        for (i, problem) in enumerate(solver.boundary_problems)
            info("Assembling boundary $i: $(problem.name)")
            append!(boundary_assembly, assemble(problem, time))
        end

        C1 = sparse(boundary_assembly.C1, dim, dim)
        C2 = sparse(boundary_assembly.C2, dim, dim)
        D = sparse(boundary_assembly.D, dim, dim)
        g = sparse(boundary_assembly.g, dim, 1)
        boundary_assembly = nothing
        gc()
        toc(timing, "boundary assembly")

        if iter == 1
            # initialize vectors in first iteration
            sol = zeros(dim)
            la = zeros(dim)
            last_sol = zeros(dim)
            last_la = zeros(dim)
        end

        tic(timing, "preprocess solution")
        # NOTE: sol and la are vectors from previous solution
        for (preprocessor, args, kwargs) in solver.linear_system_solver_preprocessors
            linear_system_solver_preprocess!(solver, iter, time, K, f, C1, C2, D, g, sol, la, Val{preprocessor}, args...; kwargs...)
        end
        toc(timing, "preprocess solution")

        gc()
        tic(timing, "solution of system")
        info("Solving linear system Ax=b")
        for (linear_solver, args, kwargs) in solver.linear_system_solvers
            last_sol = copy(sol)
            last_la = copy(la)
            sol = fill!(sol, 0.0)
            la = fill!(la, 0.0)
            linear_system_solver_solve!(solver, iter, time, K, f, C1, C2, D, g, sol, la, Val{linear_solver}, args...; kwargs...)
            # if solved only difference, add to last known solution, i.e. x(i+1) = x(i) + Δx
            solver.solve_residual && (sol += last_sol)
        end
        toc(timing, "solution of system")
        gc()

        tic(timing, "postprocess solution")
        for (postprocessor, args, kwargs) in solver.linear_system_solver_postprocessors
            linear_system_solver_postprocess!(solver, iter, time, K, f, C1, C2, D, g, sol, la, Val{postprocessor}, args...; kwargs...)
        end
        toc(timing, "postprocess solution")

        tic(timing, "update element data")
        for problem in solver.field_problems
            update!(problem, sol, Val{:elements})
        end
        for problem in solver.boundary_problems
            update!(problem, la, Val{:elements})
        end
        toc(timing, "update element data")

        toc(timing, "non-linear iteration")

        if false
            info("timing info for iteration:")
            info("boundary assembly        : ", time_elapsed(timing, "boundary assembly"))
            info("field assembly           : ", time_elapsed(timing, "field assembly"))
            info("preprocess of solution   : ", time_elapsed(timing, "preprocess solution"))
            info("solve linearized problem : ", time_elapsed(timing, "solution of system"))
            info("update element data      : ", time_elapsed(timing, "update element data"))
            info("non-linear iteration     : ", time_elapsed(timing, "non-linear iteration"))
        end

        # check convergence
        function is_converged(solver, sol, last_sol, la, last_la)
            if solver.solve_residual
                if norm(sol) < solver.nonlinear_convergence_tolerance
                    return true
                end
            else
                if abs(norm(sol) - norm(last_sol)) < solver.nonlinear_convergence_tolerance
                    return true
                end
            end
            return false
        end

        if is_converged(solver, sol, last_sol, la, last_la)
            toc(timing, "solver")
            info("converged in $iter iterations! solver finished in ", time_elapsed(timing, "solver"), " seconds.")
            return (iter, true)
        end

    end

    info("Warning: did not coverge in $(solver.nonlinear_max_iterations) iterations!")
    return (solver.nonlinear_max_iterations, false)

end
