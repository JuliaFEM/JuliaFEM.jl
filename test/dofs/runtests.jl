# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE

"""
DOF System Test Suite

Runs all tests for the DOF (Degree of Freedom) system.

Usage:
    julia --project=. test/dofs/runtests.jl
"""

using Test
using JuliaFEM
using Tensors

# All DOF types and functions are in flat JuliaFEM namespace

@testset "DOF System Tests" begin
    @testset "Field Type Unification" begin
        include("test_field_unification.jl")
    end
end
