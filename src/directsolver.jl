# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

## Direct solver

type DirectSolver <: Solver
    field_problems :: Vector{Problem}
    boundary_problems :: Vector{BoundaryProblem}
    parallel :: Bool
    nonlinear_problem :: Bool
    max_iterations :: Int64
    tol :: Float64
end

function push!(solver::DirectSolver, problem::Problem)
    push!(solver.field_problems, problem)
end

function push!(solver::DirectSolver, problem::BoundaryProblem)
    push!(solver.boundary_problems, problem)
end

""" Default initializer. """
function DirectSolver()
    DirectSolver([], [], false, true, 10, 1.0e-6)
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

""" Call solver to solve a set of problems. """
function call(solver::DirectSolver, time::Number=0.0)
    #@assert length(solver.field_problems) == 1
    info("# of field problems: $(length(solver.field_problems))")
    info("# of boundary problems: $(length(solver.boundary_problems))")
    @assert solver.nonlinear_problem == true
    
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
            if haskey(element, "reaction force")
                if !isapprox(last(element["reaction force"]).time, time)
                    push!(element["reaction force"], time => data)
                end
            else
                element["reaction force"] = (time => data)
            end
        end
    end

    toc(timing, "initialization")

    dim = 0

    for iter=1:solver.max_iterations
        info("Starting iteration $iter")
        tic(timing, "non-linear iteration")

        mapper = solver.parallel ? pmap : map

        # assemble boundary problems
        tic(timing, "boundary assembly")
        boundary_assembly = sum(mapper((p)->assemble(p, time), solver.boundary_problems))
        boundary_dofs = unique(boundary_assembly.stiffness_matrix.I)
        toc(timing, "boundary assembly")

        # assemble field problems
        # in principle if we want to static condensation we need to pass boundary dofs
        # to field problems in order to know which dofs are interior dofs and can be
        # condensated.
        tic(timing, "field assembly")
        field_assembly = sum(mapper((p)->assemble(p, time), solver.field_problems))
        field_dofs = unique(field_assembly.stiffness_matrix.I)
        info("# of dofs: $(length(field_dofs)), # of interface dofs: $(length(boundary_dofs))")
        toc(timing, "field assembly")

        tic(timing, "create sparse matrices")
        # create sparse matrices and saddle point problem
        K = sparse(field_assembly.stiffness_matrix)
        dim = size(K, 1)
        r = sparse(field_assembly.force_vector, dim, 1)
        C = sparse(boundary_assembly.stiffness_matrix, dim, dim)
        g = sparse(boundary_assembly.force_vector, dim, 1)
        A = [K C'; C spzeros(dim, dim)]
        b = [r; g]
        toc(timing, "create sparse matrices")

        tic(timing, "solution of system")
        # solve increment for linearized problem
        nz = unique(rowvals(A))  # take only non-zero rows
        sol = zeros(b)
        sol[nz] = lufact(A[nz,nz]) \ full(b[nz])
        info("solved. length of solution vector = $(length(sol))")
        toc(timing, "solution of system")
        #info(full(sol[nz]))

        tic(timing, "update element data")
        # update elements in field problems
        for field_problem in solver.field_problems
            for element in get_elements(field_problem)
                gdofs = get_gdofs(element, field_dim)
                local_sol = vec(full(sol[gdofs]))  # incremental data for element
                local_sol = reshape(local_sol, field_dim, length(element))
                local_sol = Vector{Float64}[local_sol[:,i] for i=1:length(element)]
                last(element[field_name]).data += local_sol  # <-- added
            end
        end

        # update elements in boundary problems
        for boundary_problem in solver.boundary_problems
            for element in get_elements(boundary_problem)
                gdofs = get_gdofs(element, field_dim) + dim
                local_sol = vec(full(sol[gdofs]))
                local_sol = reshape(local_sol, field_dim, length(element))
                local_sol = Vector{Float64}[local_sol[:,i] for i=1:length(element)]
                last(element["reaction force"]).data = local_sol  # <-- replaced
            end
        end
        toc(timing, "update element data")
        toc(timing, "non-linear iteration")

        if true
            info("timing info for non-linear iteration:")
            info("boundary assembly      : ", time_elapsed(timing, "boundary assembly"))
            info("field assembly         : ", time_elapsed(timing, "field assembly"))
            info("create sparse matrices : ", time_elapsed(timing, "create sparse matrices"))
            info("solution of system     : ", time_elapsed(timing, "solution of system"))
            info("update element data    : ", time_elapsed(timing, "update element data"))
            info("non-linear iteration   : ", time_elapsed(timing, "non-linear iteration"))
        end

        if norm(sol[1:dim]) < solver.tol
            toc(timing, "solver")
            info("solver finished in ", time_elapsed(timing, "solver"), " seconds.")
            return (iter, true)
        end

    end

    info("Warning: did not coverge in $(solver.max_iterations) iterations!")
    return (solver.max_iterations, false)

end

