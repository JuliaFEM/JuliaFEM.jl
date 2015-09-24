# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

abstract Problem

get_equations(pr::Problem) = pr.equations
get_dimension(pr::Type{Problem}) = nothing
get_equation(pr::Type{Problem}, el::Type{Element}) = nothing

"""
Add new element to problem
"""
function add_element!(pr::Problem, el::Element)
    eq = get_equation(typeof(pr), typeof(el))
    push!(pr.equations, eq(el))
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
