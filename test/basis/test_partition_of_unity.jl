# Test partition of unity property: sum(N) = 1
using Test
using StaticArrays, Tensors

using JuliaFEM

@testset "Partition of Unity" begin
    @testset "Seg2" begin
        N = get_basis_functions(Seg2(), Lagrange{1}(), Vec{1}((0.3,)))
        @test sum(N) ≈ 1.0
    end

    @testset "Tri3" begin
        N = get_basis_functions(Tri3(), Lagrange{1}(), Vec{2}((0.4, 0.3)))
        @test sum(N) ≈ 1.0
    end

    @testset "Quad4" begin
        N = get_basis_functions(Quad4(), Lagrange{1}(), Vec{2}((0.5, -0.5)))
        @test sum(N) ≈ 1.0
    end
end
