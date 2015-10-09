# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

abstract Problem
abstract BoundaryProblem <: Problem
abstract FieldProblem <: Problem

get_equations(pr::Problem) = pr.equations

function get_dimension(pr::Type{Problem})
    throw("Unable to determine problem dimension for problem $pr")
end

function get_equation(pr::Type{Problem}, el::Type{Element})
    throw("Could not find corresponding equation for element $el in problem $pr")
end

"""
Add new element to problem
"""
function add_element!(problem::Problem, element::Element)
    equation = get_equation(typeof(problem), typeof(element))
    push!(problem.equations, equation(element))
end

"""
Return total number of basis functions in problem
"""
function get_number_of_basis_functions(pr::Problem)
    conn = Int[]
    for eq in get_equations(pr)
        append!(conn, get_connectivity(eq))
    end
    length(unique(conn))
end

"""
Problem matrix size dimension
"""
function get_matrix_dimension(pr::Problem)
    get_dimension(typeof(pr))*get_number_of_basis_functions(pr)
end

"""
Assign global dofs for element. This doesn't do any reordering.
"""
function set_global_dofs!(pr::Problem)
    #ndim = get_dimension(pr)*get_number_of_basis_functions(pr)
    #ndim = get_matrix_dimension(pr)
    dim = get_dimension(typeof(pr))
    nconn = get_number_of_basis_functions(pr)
    ndim = dim*nconn
    Logging.debug("Problem (matrix) dimension: $ndim")
    gdofs = reshape(collect(1:ndim), dim, nconn)
    for eq in get_equations(pr)
        lconn = get_connectivity(eq)
        gconn = gdofs[:, lconn][:]
        set_global_dofs!(eq, gconn)
    end
end

function get_connectivity(pr::Problem)
    conn = Int[]
    for eq in get_equations(pr)
        el = get_element(eq)
        append!(conn, get_connectivity(el))
    end
    conn = unique(conn)
    return conn
end

"""
Calculate global dofs for equations, maybe using some bandwidth
minimizing or fill reducing algorithm
"""
function calculate_global_dofs(pr::Problem)
    conn = get_connectivity(pr)
    dim = get_dimension(typeof(pr))
    ndofs = dim*length(conn)
    Logging.debug("total dofs: $ndofs")

    mconn = maximum(conn)
    gdofs = reshape(collect(1:mconn), dim, mconn)
    dofmap = Dict{Int64, Array{Int64, 1}}()
    for (i, c) in enumerate(conn)
        dofmap[c] = gdofs[:, i]
    end
    return dofmap
end

"""
Assign global dofs for equations.
"""
function assign_global_dofs!(pr::Problem, dofmap)
    for eq in get_equations(pr)
        el = get_element(eq)
        c = get_connectivity(el)
        #gdofs = [dofmap[ci] for ci in c]
        gdofs = Int64[]
        for ci in c
            append!(gdofs, dofmap[ci])
        end
        set_global_dofs!(eq, gdofs)
    end
end

function get_lhs(pr::Problem, t::Float64)
    I = Int64[]
    J = Int64[]
    V = Float64[]
    dim = get_dimension(typeof(pr))
    for eq in filter(has_lhs, get_equations(pr))
        dofs = get_global_dofs(eq)
        lhs = integrate_lhs(eq, t)
        for (li, i) in enumerate(dofs)
            for (lj, j) in enumerate(dofs)
                push!(I, i)
                push!(J, j)
                push!(V, lhs[li, lj])
            end
        end
    end
    return I, J, V
end

function get_rhs(pr::Problem, t::Float64)
    I = Int64[]
    V = Float64[]
    dim = get_dimension(typeof(pr))
    for eq in filter(has_rhs, get_equations(pr))
        dofs = get_global_dofs(eq)
        rhs = integrate_rhs(eq, t)
        for (li, i) in enumerate(dofs)
            push!(I, i)
            push!(V, rhs[li])
        end
    end
    return I, V
end

