# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using Base.Test

@testset "add elements to problem" begin
    problem = Problem(Elasticity, "test", 2)
    element = Element(Quad4, [1, 2, 3, 4])
    elements = [element]
    add_elements!(problem, elements)
    @test problem.elements[1] == element
end
