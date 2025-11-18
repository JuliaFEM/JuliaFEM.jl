# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Physics module test suite.

Run all physics-related unit tests.
"""

using Test

@testset "Physics Module" begin
    include("test_types.jl")
    include("test_boundary_conditions.jl")
    include("test_validation.jl")
end
