# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Postprocess
using JuliaFEM.Test

@testset "extrapolate stress from gauss points to nodes" begin
    X = Dict{Int, Vector{Float64}}(
        1 => [0.0, 0.0],
        2 => [6.0, 0.0],
        3 => [6.0, 6.0],
        4 => [0.0, 6.0],
        5 => [12.0, 0.0],
        6 => [12.0, 6.0])
    el1 = Element(Quad4, [1, 2, 3, 4])
    el2 = Element(Quad4, [2, 5, 6, 3])
    el1.id = 1
    el2.id = 2
    elements = [el1, el2]
    time = 0.0
    update!(elements, "geometry", X)
    update!(get_integration_points(el1), "stress", time => [1.0, 2.0, 3.0])
    update!(get_integration_points(el2), "stress", time => [2.0, 3.0, 4.0])
    field_name = "stress"
    field_dim = 3
    calc_nodal_values!(elements, field_name, field_dim, time)
    s1 = el1("stress", [0.0, 0.0], time)
    s2 = el2("stress", [0.0, 0.0], time)
    # visually checked, see blog post "Postprocessing stress"
    @test isapprox(s1, [1.125, 2.125, 3.125])
    @test isapprox(s2, [1.875, 2.875, 3.875])
end

