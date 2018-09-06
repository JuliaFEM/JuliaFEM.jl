# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM, LinearAlgebra, Test

# compare simple 3d heat problem to analytical solution

function calc_3d_heat_model(mesh_name)
    fn = @__DIR__() * "/testdata/rod_short.med"
    mesh = aster_read_mesh(fn, mesh_name)
    problem = Problem(Heat, "rod", 1)
    bc = Problem(Dirichlet, "left support T=100", 1, "temperature")
    problem_elements = create_elements(mesh, "ROD", "FACE2")
    bc_elements = create_elements(mesh, "FACE1")
    update!(problem_elements, "thermal conductivity", 100.0)
    update!(problem_elements, "external temperature", 0.0)
    update!(problem_elements, "heat transfer coefficient", 1000.0)
    update!(bc_elements, "temperature 1", 100.0)
    add_elements!(problem, problem_elements)
    add_elements!(bc, bc_elements)
    analysis = Analysis(Linear)
    add_problems!(analysis, problem, bc)
    run!(analysis)
    time = 0.0
    T = problem("temperature", time)
    T_min = minimum(map(first, values(T)))
    return T_min
end

for model in ["Tet4", "Tet10", "Hex8", "Hex20", "Hex27"]
    Tmin = calc_3d_heat_model(model)
    Tacc = 100/3
    rtol = norm(Tmin-Tacc)/max(Tmin,Tacc)*100.0
    @debug("Temperature for model $model", model, Tmin, Tacc, rtol)
    @test isapprox(Tmin, 100/3)
end
