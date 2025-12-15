# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE

"""
Test Suite for Ciarlet Elements
"""

using Test
using JuliaFEM
using Tensors

@testset "Ciarlet Elements" begin
    @testset "CElement Type" begin
        include("test_celement_type.jl")
    end
    
    @testset "CElement Creation" begin
        include("test_celement_creation.jl")
    end
    
    @testset "CElement Interpolation" begin
        include("test_celement_interpolation.jl")
    end
end
