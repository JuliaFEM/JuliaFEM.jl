# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

function assemble!(problem::Problem, time, ::Type{Val{:mass_matrix}})
    assemble_mass_matrix!(problem, time)
end

module Abaqus
using JuliaFEM.Preprocess: create_surface_elements
end
