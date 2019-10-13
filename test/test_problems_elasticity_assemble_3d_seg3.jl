# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM, Test

# Test that if 3d continuum problem have some elements we don't know how to
# deal with, raise error with clear message

elements = [Element(Seg3, (1, 2, 3))]
problem = Problem(Elasticity, "test seg3", 3)
add_elements!(problem, elements)
@test_throws ErrorException assemble!(problem, 0.0)
