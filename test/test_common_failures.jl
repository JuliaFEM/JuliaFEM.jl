# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Testing

@testset "geometry missing" begin
    el = Element(Quad4, [1, 2, 3, 4])
    pr = Problem(Elasticity, "problem", 2)
    # this throws KeyError: geometry not found.
    # it's descriptive enough to give hint to user
    # what went wrong
    @test_throws KeyError assemble!(pr, el)
end
