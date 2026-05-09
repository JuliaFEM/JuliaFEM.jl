# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

function assemble!(problem::Problem, time, ::Type{Val{:mass_matrix}})
    assemble_mass_matrix!(problem, time)
end

function assemble!(problem::Problem, element::Element, time=0.0)
    assemble!(problem.assembly, problem, element, time)
end

module Abaqus
# Vestigial sub-module wrapper kept for downstream code that did
# `using JuliaFEM.Abaqus: create_surface_elements`. The real definition
# now lives in `JuliaFEM.Legacy.create_surface_elements`.
import ..create_surface_elements
end
