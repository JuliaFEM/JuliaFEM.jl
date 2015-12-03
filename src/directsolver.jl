# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

## Direct solver

using JuliaFEM
@everywhere using JuliaFEM
@everywhere assemble = JuliaFEM.Core.assemble

type DirectSolver <: Solver
    field_problems :: Vector{Problem}
    boundary_problems :: Vector{BoundaryProblem}
    parallel :: Bool
    nonlinear_problem :: Bool
    max_iterations :: Int64
    tol :: Float64
    dump_matrices :: Bool
    reduce_stiffness_matrix :: Bool
end

""" Default initializer. """
function DirectSolver()
    DirectSolver(
        [], # field problems
        [], # boundary problems
        false, # parallel run?
        true, # nonlinear problem?
        10, # max nonlinear iterations
        1.0e-6, # convergence tolerance
        false, # dump matrices
        true # reduce stiffness matrix
    )
end

function push!(solver::DirectSolver, problem::Problem)
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

""" Call solver to solve a set of problems. """
function call(solver::DirectSolver, time::Number=0.0)
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

        info("Assembling boundary problems...")
        tic(timing, "boundary assembly")
        boundary_assembly = sum(mapper((p)->assemble(p, time), solver.boundary_problems))
        boundary_dofs = unique(boundary_assembly.stiffness_matrix.I)
        info("# of interface dofs: $(length(boundary_dofs))")
        C = sparse(boundary_assembly.stiffness_matrix)
        g = sparse(boundary_assembly.force_vector)
        boundary_assembly = nothing
        gc()
        toc(timing, "boundary assembly")

        info("Assembling field problems...")
        dim = 0
        assemblies = []
        for (i, problem) in enumerate(solver.field_problems)
            info("Assembling body $i...")

            tic(timing, "field assembly")
            nchunks = length(workers())
            ne = length(get_elements(problem))
            kk = round(Int, collect(linspace(0, ne, nchunks+1)))
            slices = [kk[j]+1:kk[j+1] for j=1:length(kk)-1]
            field_assembly = sum(pmap((s) -> assemble(problem, s, time), slices))

            #field_assembly = assemble(problem, time)
            toc(timing, "field assembly")

            field_dofs = unique(field_assembly.stiffness_matrix.I)
            info("# of dofs in problem $i: $(length(field_dofs))")
            dim = maximum([dim, maximum(field_dofs)])
            tic(timing, "reduce stiffness matrix")
            cfield_assembly = nothing
            if solver.reduce_stiffness_matrix && (nnz(boundary) != 0)
                info("Eliminating interior dofs for body $i...")
                cfield_assembly = reduce(field_assembly, boundary_dofs)
            else
                cfield_assembly = reduce(field_assembly, boundary_dofs, Inf)
            end
            toc(timing, "reduce stiffness matrix")
            push!(assemblies, cfield_assembly)
        end

        tic(timing, "create sparse matrices")

        K = spzeros(dim, dim)
        f = spzeros(dim, 1)

        for (i, assembly) in enumerate(assemblies)
            resize!(assembly.Kc, dim, dim)
            resize!(assembly.fc, dim, 1)
            K += assembly.Kc
            f += assembly.fc
        end

        resize!(C, dim, dim)
        resize!(g, dim, 1)
        toc(timing, "create sparse matrices")

        tic(timing, "dump matrices to disk")
        if solver.dump_matrices
            save("host_$(myid())_iteration_$(iter)_matrices.jld",
                 "stiffness matrix", K, "force vector", f,
                 "constraint matrix lhs", C, "constraint matrix rhs", g)
        end
        toc(timing, "dump matrices to disk")

        all_dofs = sort(unique(rowvals(K)))
        field_dofs = setdiff(all_dofs, boundary_dofs)

        info("Solving system")
        tic(timing, "solution of system")
        sol = zeros(2*dim)
        if nnz(g) != 0
            A = [K C'; C spzeros(dim, dim)]
            b = [f; g]
            K = 0
            C = 0
            gc()
            nz = sort(unique(rowvals(A)))  # take only non-zero rows
            sol[nz] = A[nz,nz] \ full(b[nz])
        else
            K = 1/2*(K + K')
            sol[field_dofs] = cholfact(K[field_dofs, field_dofs]) \ f[field_dofs]
        end
        toc(timing, "solution of system")

        info("Solved, calculating interior dofs...")
        tic(timing, "back substitute")
        for assembly in assemblies
            length(assembly.interior_dofs) != 0 || continue
            reconstruct!(assembly, sol)
        end
        toc(timing, "back substitute")

        la = sol[dim+1:end]
        la = vec(full(la))
        sol = vec(full(sol))

        info("Problem solved. solution norm: $(norm(sol[1:dim]))")

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
            end
        end
        toc(timing, "update element data")
        toc(timing, "non-linear iteration")

        if true
            info("timing info for non-linear iteration:")
            info("boundary assembly       : ", time_elapsed(timing, "boundary assembly"))
            info("field assembly          : ", time_elapsed(timing, "field assembly"))
            info("reduce stiffness matrix : ", time_elapsed(timing, "reduce stiffness matrix"))
            info("create sparse matrices  : ", time_elapsed(timing, "create sparse matrices"))
            info("dump matrices to disk   : ", time_elapsed(timing, "dump matrices to disk"))
            info("solve problem           : ", time_elapsed(timing, "solution of system"))
            info("back substitute         : ", time_elapsed(timing, "back substitute"))
            info("update element data     : ", time_elapsed(timing, "update element data"))
            info("non-linear iteration    : ", time_elapsed(timing, "non-linear iteration"))
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

