# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

## Direct solver

using JuliaFEM
@everywhere using JuliaFEM
@everywhere assemble = JuliaFEM.Core.assemble

type DirectSolver <: Solver
    name :: ASCIIString
    field_problems :: Vector{Problem}
    boundary_problems :: Vector{BoundaryProblem}
    parallel :: Bool
    nonlinear_max_iterations :: Int64
    nonlinear_convergence_tolerance :: Float64
    linear_system_solver_preprocessors :: Vector{Symbol}
    linear_system_solvers :: Vector{Symbol}
    linear_system_solver_postprocessors :: Vector{Symbol}
end

""" Default initializer. """
function DirectSolver(name="DirectSolver")
    DirectSolver(
        name,
        [],                             # field problems
        [],                             # boundary problems
        false,                          # parallel run?
        10,                             # nonlinear problem max iterations
        5.0e-6,                         # nonlinear convergence tolerance
        Vector{Symbol}(),               # default solution preprocessors
        Vector{Symbol}([:UMFPACK]),     # linear system solver: CHOLMOD, UMFPACK
        Vector{Symbol}(),               # default solution postprocessors
    )
end

function set_name!(solver::DirectSolver, name::ASCIIString)
    solver.name = name
end

function set_linear_system_solver!(solver::DirectSolver, method::Symbol)
    solver.linear_system_solvers = Vector{Symbol}([method])
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

""" Call solver to solve a set of problems. """
function call(solver::DirectSolver, time::Real=0.0)
    info("Starting solver $(solver.name)")
    info("# of field problems: $(length(solver.field_problems))")
    info("# of boundary problems: $(length(solver.boundary_problems))")
    (length(solver.field_problems) != 0) || error("no field problems defined for solver, use push!(solver, problem, ...) to define field problems.")

    timing = Dict{ASCIIString, Float64}()
    tic(timing, "solver")
    tic(timing, "initialization")

    # check that all problems are "same kind"
    field_name = get_unknown_field_name(solver.field_problems[1])
    field_dim = get_unknown_field_dimension(solver.field_problems[1])
    for field_problem in solver.field_problems
        get_unknown_field_name(field_problem) == field_name || error("several different fields not supported yet")
        get_unknown_field_dimension(field_problem) == field_dim || error("several different field dimensions not supported yet")
    end

    # create initial fields for this increment
    # i.e., copy last known values as initial guess
    # for this increment

    for field_problem in solver.field_problems
        for element in get_elements(field_problem)
            gdofs = get_gdofs(element, field_dim)
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

    for boundary_problem in solver.boundary_problems
        for element in get_elements(boundary_problem)
            gdofs = get_gdofs(element, field_dim)
            data = Vector{Float64}[zeros(field_dim) for i in 1:length(element)]

            # add new field "reaction force" for boundary element if not found
            if haskey(element, "reaction force")
                if !isapprox(last(element["reaction force"]).time, time)
                    push!(element["reaction force"], time => data)
                end
            else
                element["reaction force"] = (time => data)
            end

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

    toc(timing, "initialization")

    dim = nothing
    sol = nothing
    la = nothing

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
        end

        tic(timing, "preprocess solution")
        # NOTE: sol and la are vectors from previous solution
        for preprocessor in solver.linear_system_solver_preprocessors
            linear_system_solver_preprocess!(solver, iter, time, K, f, C1, C2, D, g, sol, la, Val{preprocessor})
        end
        toc(timing, "preprocess solution")

        gc()
        tic(timing, "solution of system")
        info("Solving linear system Ax=b")
        for linear_solver in solver.linear_system_solvers
            linear_system_solver_solve!(solver, iter, time, K, f, C1, C2, D, g, sol, la, Val{linear_solver})
        end
        toc(timing, "solution of system")
        gc()

        tic(timing, "postprocess solution")
        for postprocessor in solver.linear_system_solver_postprocessors
            linear_system_solver_postprocess!(solver, iter, time, K, f, C1, C2, D, g, sol, la, Val{postprocessor})
        end
        toc(timing, "postprocess solution")

        tic(timing, "update element data")
        # update elements in field problems
        for field_problem in solver.field_problems
            for element in get_elements(field_problem)
                gdofs = get_gdofs(element, field_dim)
                local_sol = sol[gdofs]  # incremental data for element
                local_sol = reshape(local_sol, field_dim, length(element))
                local_sol = Vector{Float64}[local_sol[:,i] for i=1:length(element)]
                last(element[field_name]).data += local_sol  # <-- added
            end
        end

        # update elements in boundary problems
        for boundary_problem in solver.boundary_problems
            for element in get_elements(boundary_problem)
                gdofs = get_gdofs(element, field_dim)
                local_sol = la[gdofs]
                local_sol = reshape(local_sol, field_dim, length(element))
                local_sol = Vector{Float64}[local_sol[:,i] for i=1:length(element)]
                last(element["reaction force"]).data = local_sol  # <-- replaced
                # FIXME: Quick and dirty, updating dirichlet boundary problem
                # is causing drifting and convergence issue
                if typeof(boundary_problem) <: BoundaryProblem{DirichletProblem}
#                    info("skipping dirichlet problem update")
                    continue
                end
                primary_sol = sol[gdofs]  # solution of primary field
                primary_sol = reshape(primary_sol, field_dim, length(element))
                primary_sol = Vector{Float64}[primary_sol[:,i] for i=1:length(element)]
                last(element[field_name]).data = primary_sol
            end
        end
        toc(timing, "update element data")

        toc(timing, "non-linear iteration")

        if true
            info("timing info for iteration:")
            info("boundary assembly        : ", time_elapsed(timing, "boundary assembly"))
            info("field assembly           : ", time_elapsed(timing, "field assembly"))
            info("preprocess of solution   : ", time_elapsed(timing, "preprocess solution"))
            info("solve linearized problem : ", time_elapsed(timing, "solution of system"))
            info("update element data      : ", time_elapsed(timing, "update element data"))
            info("non-linear iteration     : ", time_elapsed(timing, "non-linear iteration"))
        end

        if (norm(sol) < solver.nonlinear_convergence_tolerance)
            toc(timing, "solver")
            info("converged in $iter iterations! solver finished in ", time_elapsed(timing, "solver"), " seconds.")
            return (iter, true)
        end

    end

    info("Warning: did not coverge in $(solver.nonlinear_max_iterations) iterations!")
    return (solver.nonlinear_max_iterations, false)

end
