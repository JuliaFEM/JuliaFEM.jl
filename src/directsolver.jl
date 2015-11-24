# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

## Direct solver

type DirectSolver <: Solver
    field_problems :: Vector{FieldProblem}
    boundary_problems :: Vector{BoundaryProblem}
    parallel :: Bool
    nonlinear_problem :: Bool
    max_iterations :: Int64
    tol :: Float64
end

function push!(solver::DirectSolver, problem::FieldProblem)
    push!(solver.field_problems, problem)
end

function push!(solver::DirectSolver, problem::BoundaryProblem)
    push!(solver.boundary_problems, problem)
end

""" Default initializer. """
function DirectSolver()
    DirectSolver([], [], false, true, 10, 1.0e-6)
end

""" Call solver to solve a set of problems. """
function call(solver::DirectSolver, time::Number=0.0)
    #@assert length(solver.field_problems) == 1
    info("# of field problems: $(length(solver.field_problems))")
    info("# of boundary problems: $(length(solver.boundary_problems))")
    @assert solver.nonlinear_problem == true

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
        for equation in get_equations(field_problem)
            element = get_element(equation)
            gdofs = get_gdofs(field_problem, equation)
            if !isapprox(last(element[field_name]).time, time)
                last_data = copy(last(element[field_name]).data)
                push!(element[field_name], time => last_data)
            end
        end
    end

    for boundary_problem in solver.boundary_problems
        for equation in get_equations(boundary_problem)
            element = get_element(equation)
            gdofs = get_gdofs(boundary_problem, equation)
            eqdim = size(equation)[2]
            data = Vector{Float64}[zeros(field_dim) for i in 1:eqdim]
            if !isapprox(last(element["reaction force"]).time, time)
                push!(element["reaction force"], time => data)
            end
        end
    end

    dim = 0

    for iter=1:solver.max_iterations
        tic()
        info("Starting iteration $iter")

        mapper = solver.parallel ? pmap : map

        # assemble boundary problems
        boundary_assembly = sum(mapper((p)->assemble(p, time), solver.boundary_problems))
        boundary_dofs = unique(boundary_assembly.stiffness_matrix.I)

        # assemble field problems
        # in principle if we want to static condensation we need to pass boundary dofs
        # to field problems in order to know which dofs are interior dofs and can be
        # condensated.
        field_assembly = sum(mapper((p)->assemble(p, time), solver.field_problems))
        field_dofs = unique(field_assembly.stiffness_matrix.I)
        info("# of dofs: $(length(field_dofs)), # of interface dofs: $(length(boundary_dofs))")

        # create sparse matrices and saddle point problem
        K = sparse(field_assembly.stiffness_matrix)
        dim = size(K, 1)
        r = sparse(field_assembly.force_vector, dim, 1)
        C = sparse(boundary_assembly.stiffness_matrix, dim, dim)
        g = sparse(boundary_assembly.force_vector, dim, 1)
        A = [K C'; C spzeros(dim, dim)]
        b = [r; g]

        # solve increment for linearized problem
        nz = unique(rowvals(A))  # take only non-zero rows
        sol = zeros(b)
        sol[nz] = lufact(A[nz,nz]) \ full(b[nz])
        info("solved. length of solution vector = $(length(sol))")
        #info(full(sol[nz]))

        # update elements in field problems
        for field_problem in solver.field_problems
            for equation in get_equations(field_problem)
                element = get_element(equation)
                gdofs = get_gdofs(field_problem, equation)
                eqsize = size(equation)
                local_sol = vec(full(sol[gdofs]))  # incremental data for element
                local_sol = reshape(local_sol, eqsize)
                local_sol = Vector{Float64}[local_sol[:,i] for i=1:size(local_sol,2)]
                last(element[field_name]).data += local_sol  # <-- added
            end
        end

        # update elements in boundary problems
        for boundary_problem in solver.boundary_problems
            for equation in get_equations(boundary_problem)
                element = get_element(equation)
                gdofs = get_gdofs(boundary_problem, equation) + dim
                eqsize = size(equation)
                local_sol = vec(full(sol[gdofs]))
                #info("local sol = $local_sol")
                local_sol = reshape(local_sol, field_dim, eqsize[2])
                local_sol = Vector{Float64}[local_sol[:,i] for i=1:size(local_sol,2)]
                last(element["reaction force"]).data = local_sol  # <-- replaced
            end
        end

        info("Iteration took $(toq()) seconds")

        if norm(sol[1:dim]) < solver.tol
            return (iter, true)
        end

    end

    info("Warning: did not coverge in $(solver.max_iterations) iterations!")
    return (solver.max_iterations, false)

end

