# Test basis function derivatives
using Test
using StaticArrays, Tensors

using JuliaFEM

@testset "Derivatives" begin
    @testset "Seg2 constant gradient" begin
        dN = get_basis_derivatives(Seg2(), Lagrange{1}(), Vec{1}((0.0,)))
        @test dN[1][1] ≈ -0.5
        @test dN[2][1] ≈ 0.5
    end

    @testset "Tri3 constant gradients" begin
        dN = get_basis_derivatives(Tri3(), Lagrange{1}(), Vec{2}((0.3, 0.4)))
        @test dN[1] ≈ Vec{2}((-1.0, -1.0))
        @test dN[2] ≈ Vec{2}((1.0, 0.0))
        @test dN[3] ≈ Vec{2}((0.0, 1.0))
    end
end
