# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# Functions to handle global assembly of problem

""" Global assembly. """
type GlobalAssembly <: Assembly
    ndofs :: Int
    mass_matrix :: SparseMatrixCSC
    stiffness_matrix :: SparseMatrixCSC
    force_vector :: SparseMatrixCSC
end

""" Initialize global assembly of size ndofs. """
function initialize_global_assembly(ndofs::Int=1)
    mass_matrix = spzeros(ndofs, ndofs)
    stiffness_matrix = spzeros(ndofs, ndofs)
    force_vector = spzeros(ndofs, 1)
    return GlobalAssembly(ndofs, mass_matrix, stiffness_matrix, force_vector)
end

""" Initialize global assembly, get dimension from problem. """
function initialize_global_assembly(problem::Problem)
    dim, ndofs = size(problem)
    return initialize_global_assembly(ndofs)
end

""" Initialize or empty workspace for global assembly. """
function initialize_global_assembly!(assembly::GlobalAssembly, problem::Problem)
    ndofs = prod(size(problem))
    if ndofs != assembly.ndofs
        # if problem size changes, automatically initialize new work space
        assembly.ndofs = ndofs
        assembly.mass_matrix = spzeros(ndofs, ndofs)
        assembly.stiffness_matrix = spzeros(ndofs, ndofs)
        assembly.force_vector = spzeros(ndofs, 1)
        return
    end
    # otherwise, empty workspace ready for next iteration
    fill!(assembly.mass_matrix, 0.0)
    fill!(assembly.stiffness_matrix, 0.0)
    fill!(assembly.force_vector, 0.0)
    return
end

""" Calculate global assembly for a problem. """
function calculate_global_assembly!(assembly::GlobalAssembly, problem::Problem, time::Number=Inf)

    unknown_field_name = get_unknown_field_name(problem)
    initialize_global_assembly!(assembly, problem) # zero all
    dim, ndofs = size(problem)
    Logging.info("assembling problem for $unknown_field_name")
    Logging.info("dimension of unknown field: $dim, problem dofs: $ndofs")
    local_assembly = initialize_local_assembly()
    for (i, equation) in enumerate(get_equations(problem))
        calculate_local_assembly!(local_assembly, equation, unknown_field_name, time)
        conn = get_connectivity(get_element(equation))
        gdofs = get_gdofs(problem, equation)
        assembly.mass_matrix[gdofs, gdofs] += local_assembly.mass_matrix
        assembly.stiffness_matrix[gdofs, gdofs] += local_assembly.stiffness_matrix
        assembly.force_vector[gdofs] += local_assembly.force_vector
    end
end
