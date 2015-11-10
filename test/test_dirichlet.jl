
# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

module TestAutoDiffWeakForm

using JuliaFEM.Test
using JuliaFEM
using JuliaFEM: Seg2, DirichletProblem

function test_dirichlet_problem()
    element = Seg2([3, 4])
    element["geometry"] = Vector[[1.0, 1.0], [0.0, 1.0]]
    problem = DirichletProblem(1)
    push!(problem, element)
end

end
