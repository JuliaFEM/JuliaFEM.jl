# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

#=
""" Solution preprocessor: dump matrices to disk before solution.

Examples
--------
julia> solver = DirectSolver()
julia> push!(solver.linear_system_solver_preprocessors, :dump_matrices)

"""
function linear_system_solver_preprocess!(solver, iter, time, K, f, C1, C2, D, g, sol, la, ::Type{Val{:dump_matrices}})
    filename = "matrices_$(solver.name)_host_$(myid())_iteration_$(iter).jld"
    info("dumping matrices to disk, file = $filename")
    save(filename,
        "stiffness matrix K", K,
        "force vector f", f,
        "constraint matrix C1", C1,
        "constraint matrix C2", C2,
        "constraint matrix D", D,
        "constraint vector g", g)
end

function linear_system_solver_postprocess!
end
=#


