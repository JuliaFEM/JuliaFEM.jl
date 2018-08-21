# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM, Test

# test postprocessing of strain and stress

X = Dict(
    1 => [0.0, 0.0, 0.0],
    2 => [1.0, 0.0, 0.0],
    3 => [1.0, 1.0, 0.0],
    4 => [0.0, 1.0, 0.0],
    5 => [0.0, 0.0, 1.0],
    6 => [1.0, 0.0, 1.0],
    7 => [1.0, 1.0, 1.0],
    8 => [0.0, 1.0, 1.0])

u = Dict(
    1 => [0.0, 0.0, 0.0],
    2 => [-1/3, 0.0, 0.0],
    3 => [-1/3, -1/3, 0.0],
    4 => [0.0, -1/3, 0.0],
    5 => [0.0, 0.0, 1.0],
    6 => [-1/3, 0.0, 1.0],
    7 => [-1/3, -1/3, 1.0],
    8 => [0.0, -1/3, 1.0])

element = Element(Hex8, (1, 2, 3, 4, 5, 6, 7, 8))
update!(element, "geometry", 0.0 => X)
update!(element, "displacement", 0.0 => u)
update!(element, "youngs modulus", 288.0)
update!(element, "poissons ratio", 1/3)

body = Problem(Elasticity, "[0,1]³ elastic unit block", 3)
add_elements!(body, element)
postprocess!(body, 0.0, Val{:strain})
postprocess!(body, 0.0, Val{:stress})

geom = element("geometry", 0.0)
strain = element("strain", 0.0)
stress = element("stress", 0.0)

for i in 1:length(geom)
    @test isapprox(strain[i], [-1/3, -1/3, 1.0, 0.0, 0.0, 0.0])
    @test isapprox(stress[i], [0.0, 0.0, 288.0, 0.0, 0.0, 0.0])
end
