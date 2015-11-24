# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# Functions to handle global assembly of problem

function assemble!(assembly::Assembly, problem::Problem, time::Number=0.0, empty_assembly::Bool=true)
    if empty_assembly
        empty!(assembly)
    end
    for equation in get_equations(problem)
        assemble!(assembly, equation, time, problem)
    end
end

function assemble(problem::Problem, time::Number=0.0)
    assembly = Assembly()
    for equation in get_equations(problem)
        assemble!(assembly, equation, time, problem)
    end
    return assembly
end

function Base.(:+)(ass1::Assembly, ass2::Assembly)
    mass_matrix = ass1.mass_matrix + ass2.mass_matrix
    stiffness_matrix = ass1.stiffness_matrix + ass2.stiffness_matrix
    force_vector = ass1.force_vector + ass2.force_vector
    return Assembly(mass_matrix, stiffness_matrix, force_vector)
end
