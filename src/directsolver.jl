# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

## Direct solver

type DirectSolver <: Solver
    field_problems :: Vector{FieldProblem}
    boundary_problems :: Vector{BoundaryProblem}
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
    DirectSolver([], [], true, 10, 1.0e-6)
end

""" Call solver to solve a set of problems. """
function call(solver::DirectSolver, time::Number=0.0)
    @assert length(solver.field_problems) == 1
    @assert length(solver.boundary_problems) == 1
    @assert solver.nonlinear_problem == true

    problem1 = solver.field_problems[1]
    problem2 = solver.boundary_problems[1]

    x = zeros(3)
    dx = zeros(3)
    dims = nothing

    for iter=1:solver.max_iterations
        tic()
        info("Starting iteration $iter")
        assembly1 = Assembly()
        assemble!(assembly1, problem1, time)
        assembly2 = Assembly()
        assemble!(assembly2, problem2, time)

        A1 = sparse(assembly1.stiffness_matrix)
        dims = size(A1)
        b1 = sparse(assembly1.force_vector, dims[1], 1)
        A2 = sparse(assembly2.stiffness_matrix, dims[1], dims[2])
        b2 = sparse(assembly2.force_vector, dims[1], 1)
    
        # create a saddle point problem
        A = [A1 A2; A2' zeros(A2)]
        b = [b1; b2]

        if length(b) != length(x)
            info("iter $iter: resizing solution vector")
            resize!(x, length(b))
            resize!(dx, length(b))
            fill!(x, 0.0)
            fill!(dx, 0.0)
        end

        # solve problem, update solution vector
        nz = unique(rowvals(A))  # take only non-zero rows
        dx[nz] = lufact(A[nz,nz]) \ full(b[nz])
        x += dx

        # get "problem-wise" solution vectors
        x1 = x[1:dims[1]]
        x2 = x[dims[1]+1:end]

        # update field for elements in problem 1
        for equation in get_equations(problem1)
            element = get_element(equation)
            field_name = get_unknown_field_name(problem1)
            gdofs = get_gdofs(problem1, equation)
            local_sol = vec(full(x1[gdofs]))
            eqsize = size(equation)
            if eqsize[1] != 1
                local_sol = reshape(local_sol, eqsize)
            end
            #info("problem1: pushing to $field_name")
            push!(element[field_name], time => local_sol)
        end

        # update field for elements in problem 2 (Dirichlet boundary)
        for equation in get_equations(problem2)
            element = get_element(equation)
            field_name = "reaction force" #get_unknown_field_name(problem2)
            gdofs = get_gdofs(problem2, equation)
            local_sol = vec(full(x1[gdofs]))
            eqsize = size(equation)
            if eqsize[1] != 1
                local_sol = reshape(local_sol, eqsize)
            end
            #info("problem2: pushing to $field_name")
            push!(element[field_name], time => local_sol)
        end
        
        if norm(dx[1:dims[1]]) < solver.tol
            return (iter, true)
        end

        info("Iteration took $(toq()) seconds")
    end

    info("Warning: did not coverge in $(solver.max_iterations) iterations!")
    return (solver.max_iterations, false)

end

