# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

#= Solution norms for piston model

piston_19611_P2.inp     iter 1      2.048090408266966

=#

## Direct solver

using JuliaFEM
@everywhere using JuliaFEM
@everywhere assemble = JuliaFEM.Core.assemble

type DirectSolver <: Solver
    name :: ASCIIString
    field_problems :: Vector{Problem}
    boundary_problems :: Vector{BoundaryProblem}
    parallel :: Bool
    nonlinear_problem :: Bool
    max_iterations :: Int64
    tol :: Float64
    dump_matrices :: Bool
    reduce_stiffness_matrix :: Bool
    method :: Symbol
end

""" Default initializer. """
function DirectSolver(name="DirectSolver")
    DirectSolver(
        name,
        [], # field problems
        [], # boundary problems
        false, # parallel run?
        true, # nonlinear problem?
        10, # max nonlinear iterations
        1.0e-6, # convergence tolerance
        false, # dump matrices
        true, # reduce stiffness matrix
        :CHOLMOD # method: CHOLMOD, UMFPACK, PETSc_GMRES
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


"""
Solve problem

    Ku + C'λ = f
    Cu       = g

"""
function solve(K, f, C, g, ::Type{Val{:CHOLMOD}})

    t0 = time()
    # make sure K is symmetric
#   K = Symmetric(K)
    s = maximum(abs(1/2*(K + K') - K))
    @assert s < 1.0e-6
    K = 1/2*(K + K')

    dim = size(K, 1)

    # make sure C is square
    boundary_dofs = unique(rowvals(C))
    boundary_dofs2 = unique(rowvals(C'))
    @assert length(boundary_dofs) == length(boundary_dofs2)
    @assert setdiff(Set(boundary_dofs), Set(boundary_dofs2)) == Set()

    all_dofs = unique(rowvals(K))
    interior_dofs = setdiff(all_dofs, boundary_dofs)
    info("CHOLMOD: all dofs = $(length(all_dofs))")
    info("CHOLMOD: interior dofs = $(length(interior_dofs))")
    info("CHOLMOD: boundary dofs = $(length(boundary_dofs))")

    # solve displacement on known boundary
    LUF = lufact(C[boundary_dofs, boundary_dofs])
    u = zeros(dim)
    u[boundary_dofs] = LUF \ full(g[boundary_dofs])
    info("CHOLMOD: displacement on boundary solved.")
    normub = norm(u[boundary_dofs])
    if isapprox(normub, 0.0)
        info("CHOLMOD: homogeneous dirichlet boundary")
    end

    # factorize interior domain using cholmod
    t = time()
    CF = cholfact(K[interior_dofs, interior_dofs])
    Kib = K[interior_dofs, boundary_dofs]
    Kbb = K[boundary_dofs, boundary_dofs]
    fi = f[interior_dofs]
    info("CHOLMOD: LDLt factorization done in ", time()-t, " seconds")

    # solve interior domain + lagrange multipliers
    u[interior_dofs] = CF \ (fi - Kib*u[boundary_dofs])
    la = zeros(dim)
    la[boundary_dofs] = LUF \ full(Kib'*u[interior_dofs] - Kbb*u[boundary_dofs])
    info("CHOLMOD: solved in ", time()-t0, " seconds. norm = ", norm(u))
    return u, la
end

function solve(K, f, C, g, ::Type{Val{:UMFPACK}})
    t0 = time()
    dim = size(K, 1)
    A = nothing
    try
        A = [K C'; C spzeros(dim, dim)]
    catch
        info("UMFPACK: size(K) = ", size(K))
        info("UMFPACK: size(C) = ", size(C))
        error("UMFPACK: Failed to construct problem. dim = $dim")
    end
    b = [f; g]
    nz = sort(unique(rowvals(A)))
    u = zeros(length(b))
    u[nz] = lufact(A[nz,nz]) \ full(b[nz])
    info("UMFPACK: solved in ", time()-t0, " seconds. norm = ", norm(u[1:dim]))
    return u[1:dim], u[dim+1:end]
end


""" Call solver to solve a set of problems. """
function call(solver::DirectSolver, time::Number=0.0)
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

        tic(timing, "field assembly")
        info("Assembling field problems...")
        field_assembly = Assembly()
        for (i, problem) in enumerate(solver.field_problems)
            info("Assembling body $i: $(problem.name)")
            append!(field_assembly, assemble(problem, time))
        end
        K = sparse(field_assembly.stiffness_matrix)
        dim = size(K, 1)
        info("dim = $dim")
        f = sparse(field_assembly.force_vector, dim, 1)
        field_assembly = nothing
        gc()
        toc(timing, "field assembly")

        tic(timing, "boundary assembly")
        info("Assembling boundary problems...")
        boundary_assembly = Assembly()
        for (i, problem) in enumerate(solver.boundary_problems)
            info("Assembling boundary $i: $(problem.name)")
            append!(boundary_assembly, assemble(problem, time))
        end
        C = sparse(boundary_assembly.stiffness_matrix, dim, dim)
        g = sparse(boundary_assembly.force_vector, dim, 1)
        boundary_assembly = nothing
        gc()
        toc(timing, "boundary assembly")


#       resize!(C, dim, dim)
#       resize!(g, dim, 1)
#       resize!(f, dim, 1)

        tic(timing, "dump matrices to disk")
        if solver.dump_matrices
            filename = "matrices_$(solver.name)_host_$(myid())_iteration_$(iter).jld"
            info("dumping matrices to disk, file = $filename")
            save(filename, "stiffness matrix", K, "force vector", f,
                 "constraint matrix lhs", C, "constraint matrix rhs", g)
        end
        toc(timing, "dump matrices to disk")

        tic(timing, "solution of system")
        info("Solving system")
        gc()
#       whos()
        sol, la = solve(K, f, C, g, Val{solver.method})
        gc()
        toc(timing, "solution of system")

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
            info("timing info for iteration:")
            info("boundary assembly       : ", time_elapsed(timing, "boundary assembly"))
            info("field assembly          : ", time_elapsed(timing, "field assembly"))
            info("dump matrices to disk   : ", time_elapsed(timing, "dump matrices to disk"))
            info("solve problem           : ", time_elapsed(timing, "solution of system"))
            info("update element data     : ", time_elapsed(timing, "update element data"))
            info("non-linear iteration    : ", time_elapsed(timing, "non-linear iteration"))
        end

        if (norm(sol) < solver.tol) || !solver.nonlinear_problem
            toc(timing, "solver")
            info("solver finished in ", time_elapsed(timing, "solver"), " seconds.")
            return (iter, true)
        end

    end

    info("Warning: did not coverge in $(solver.max_iterations) iterations!")
    return (solver.max_iterations, false)

end
