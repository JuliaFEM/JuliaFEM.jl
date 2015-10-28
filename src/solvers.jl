# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# Solver stuff

abstract Solver

"""
Solve field equations for single element with some dofs fixed. This can be used
to test nonlinear element formulations.
"""
function solve!(equation::Equation, unknown_field_name::ASCIIString,
                 free_dofs::Array{Int, 1}, time::Number=Inf; 
                 max_iterations::Int=10, tolerance::Float64=1.0e-12, dump_matrices::Bool=false)
    element = get_element(equation)
    x0 = element[unknown_field_name](-Inf)
    x = zeros(prod(size(equation)))
    dx = fill!(similar(x), 0.0)
    la = initialize_local_assembly()
    for i=1:max_iterations
        calculate_local_assembly!(la, equation, unknown_field_name)
        A = la.stiffness_matrix[free_dofs, free_dofs]
        b = la.force_vector[free_dofs]
        if dump_matrices
            dump(full(A))
            dump(full(b)')
        end
        dx[free_dofs] = A \ b
        x += dx
        new_field = similar(x0, x)
        new_field.time = time
        new_field.increment = i
        push!(element[unknown_field_name], new_field)
        if norm(dx) < tolerance
            return
        end
    end
    Logging.err("Did not converge in $max_iterations iterations")
end

"""
Solve field equations for a single problem with some dofs fixed. This can be used
to test nonlinear element formulations. Dirichlet boundary is assumed to be homogeneous
and degrees of freedom are eliminated. So if boundary condition is known in nodal
points and everything is zero this should be quite good.
"""
function solve!(problem::Problem, free_dofs::Array{Int, 1}, time::Number=Inf;
                max_iterations::Int=10, tolerance::Float64=1.0e-12, dump_matrices::Bool=false)
    ga = initialize_global_assembly(problem)
    x = zeros(ga.ndofs)
    dx = fill!(similar(x), 0.0)
    field_name = get_unknown_field_name(problem)
    dim = get_unknown_field_dimension(problem)
    for i=1:max_iterations
        calculate_global_assembly!(ga, problem)
        A = ga.stiffness_matrix[free_dofs, free_dofs]
        b = ga.force_vector[free_dofs]
        if dump_matrices
            dump(full(A))
            dump(full(b)')
        end
        dx[free_dofs] = lufact(A) \ full(b)
        x += dx
        for equation in get_equations(problem)
            element = get_element(equation)
            conn = get_connectivity(element)
            gdofs = vec(vcat([dim*conn'-i for i=dim-1:-1:0]...))
            old_field = element[field_name](Inf)
            new_field = similar(old_field, full(x[gdofs]))
            new_field.time = time
            new_field.increment = i
            push!(element[field_name], new_field)
        end
        if norm(dx) < tolerance
            return
        end
    end
    Logging.err("Did not converge in $max_iterations iterations")
end

""" Add new problem to solver. """
function add_problem!(solver::Solver, problem::Problem)
    push!(solver.problems, problem)
end

function Base.push!(solver::Solver, problem::Problem)
    push!(solver.problems, problem)
end

""" Get all problems assigned to solver. """
function get_problems(s::Solver)
    return s.problems
end

## SimpleSolver -- tiny direct demo solver
""" Simple solver for educational purposes. """
type SimpleSolver <: Solver
    problems
end

""" Default initializer. """
function SimpleSolver()
    SimpleSolver(Problem[])
end

"""
Call solver to solve a set of problems.

This is a simple direct solver for demonstration purposes. It handles the
common situation, i.e., some main field problem and it's Dirichlet boundary.

    Au + C'λ = f
    Cu       = g

"""
function call(solver::SimpleSolver, time::Number=Inf)
    p1, p2 = get_problems(solver)

    ga1 = initialize_global_assembly(p1)
    calculate_global_assembly!(ga1, p1)
    ga2 = initialize_global_assembly(p2)
    calculate_global_assembly!(ga2, p2)

    A1 = ga1.stiffness_matrix
    b1 = ga1.force_vector
    A2 = ga2.stiffness_matrix
    b2 = ga2.force_vector
    
    # create a saddle point problem
    A = [A1 A2; A2' zeros(A2)]
    b = [b1; b2]

    # solve problem
    nz = unique(rowvals(A))  # here we remove any zero rows
    x = zeros(b)
    x[nz] = lufact(A[nz,nz]) \ full(b[nz])

    # get "problem-wise" solution vectors
    x1 = x[1:length(b1)]
    x2 = x[length(b1)+1:end]

    # update field for elements in problem 1
    for equation in get_equations(p1)
        element = get_element(equation)
        field_name = get_unknown_field_name(p1)
        gdofs = get_gdofs(p1, equation)
        element_solution = full(x1[gdofs])
        field = Field(time, element_solution)
        push!(element[field_name], field)
    end

    # update field for elements in problem 2 (Dirichlet boundary)
    for equation in get_equations(p2)
        element = get_element(equation)
        field_name = get_unknown_field_name(p2)
        gdofs = get_gdofs(p2, equation)
        element_solution = full(x2[gdofs])
        field = Field(time, element_solution)
        push!(element[field_name], field)
    end
end

