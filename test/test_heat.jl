# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM, LinearAlgebra, Test

# compare simple 3d heat problem to code aster solution

function calc_3d_heat_model(mesh_name)
    fn = @__DIR__() * "/testdata/rod_short.med"
    mesh = aster_read_mesh(fn, mesh_name)
    element_sets = join(keys(mesh.element_sets), ", ")
    @debug("element sets: $element_sets")
    # x -> FACE1 ... FACE2
    # y -> FACE3 ... FACE4
    # z -> FACE5 ... FACE6
    # rod has longer dimension in x direction, first face comes
    # first in corresponding axis direction
    rod = Problem(Heat, "rod", 1)
    rod_elements = create_elements(mesh, "ROD")
    face2 = create_elements(mesh, "FACE2")
    face3 = create_elements(mesh, "FACE3")
    face4 = create_elements(mesh, "FACE4")
    face5 = create_elements(mesh, "FACE5")
    face6 = create_elements(mesh, "FACE6")
    update!(rod_elements, "thermal conductivity", 50.0)
    update!(face2, "external temperature", 20.0)
    update!(face2, "heat transfer coefficient", 60.0)
    update!(face3, "external temperature", 30.0)
    update!(face3, "heat transfer coefficient", 50.0)
    update!(face4, "external temperature", 40.0)
    update!(face4, "heat transfer coefficient", 40.0)
    update!(face5, "external temperature", 50.0)
    update!(face5, "heat transfer coefficient", 30.0)
    update!(face6, "external temperature", 60.0)
    update!(face6, "heat transfer coefficient", 20.0)
    push!(rod, rod_elements, face2, face3, face4, face5, face6)
    bc = Problem(Dirichlet, "left support T=100", 1, "temperature")
    bc_elements = create_elements(mesh, "FACE1")
    update!(bc_elements, "temperature 1", 100.0)
    add_elements!(bc, bc_elements)
    analysis = Analysis(Linear)
    add_problems!(analysis, rod, bc)
    run!(analysis)
    time = 0.0
    temperature = rod("temperature", time)
    minimum_temperature = minimum(values(temperature))
    return minimum_temperature
end

CA_sol = Dict(
    "Tet4" => 3.01872246268290E+01,
    "Hex8" => 3.01263406641066E+01,
    "Tet10" => 4.38924023356612E+01,
    "Hex20" => 4.57539800177123E+01,
    "Hex27" => 4.57760386068096E+01)

models = ["Tet4", "Hex8", "Hex20", "Hex27", "Tet10"]

for model in models
    T_min = calc_3d_heat_model(model)
    T_ca = CA_sol[model]
    rtol = norm(T_min-T_ca)/max(T_min,T_ca)*100.0
    @debug("Results for model $model", model, T_min, T_ca, rtol)
    @test rtol < 1.0e-9
end
