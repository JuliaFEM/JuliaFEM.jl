# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# Functions to handle global assembly of problem

function assemble!(assembly::Assembly, problem::Problem, time::Number=0.0)
    empty!(assembly)
    for equation in get_equations(problem)
        assemble!(assembly, equation, time, problem)
    end
end
