# Test interpolation with basis functions
using Test
using StaticArrays, Tensors

using JuliaFEM

@testset "Interpolation" begin
    @testset "Seg2 linear" begin
        values = SVector(1.0, 2.0)
        N = get_basis_functions(Seg2(), Lagrange{1}(), Vec{1}((0.0,)))
        result = dot(values, N)
        @test result ≈ 1.5
    end

    @testset "Tri3 at node" begin
        values = SVector(1.0, 2.0, 3.0)
        N = get_basis_functions(Tri3(), Lagrange{1}(), Vec{2}((1.0, 0.0)))
        result = dot(values, N)
        @test result ≈ 2.0
    end
end
