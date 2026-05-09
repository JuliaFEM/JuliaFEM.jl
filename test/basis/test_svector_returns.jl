# Test that basis functions return SVector types
using Test
using StaticArrays, Tensors

using JuliaFEM

@testset "SVector Return Types" begin
    @testset "Basis functions" begin
        N = get_basis_functions(Tri3(), Lagrange{1}(), Vec{2}((0.5, 0.25)))
        @test N isa SVector{3,Float64}
    end

    @testset "Derivatives" begin
        dN = get_basis_derivatives(Tri3(), Lagrange{1}(), Vec{2}((0.5, 0.25)))
        @test dN isa SVector{3,Vec{2,Float64}}
    end
end
