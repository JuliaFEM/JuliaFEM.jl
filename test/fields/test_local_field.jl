# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using Test
using JuliaFEM
using Tensors

@testset "LocalField" begin
    @testset "Scalar field (Temperature)" begin
        # Create temperature field at a point
        T = 300.0
        ∇T = Vec{3}((1.0, 2.0, 3.0))
        Ṫ = 0.5
        ∇Ṫ = Vec{3}((0.1, 0.2, 0.3))

        T_local = LocalField(T, ∇T, Ṫ, ∇Ṫ)

        @test T_local.value == 300.0
        @test T_local.gradient == Vec{3}((1.0, 2.0, 3.0))
        @test T_local.rate == 0.5
        @test T_local.gradient_rate == Vec{3}((0.1, 0.2, 0.3))

        # Type stability
        @test typeof(T_local) == LocalField{Float64, Vec{3,Float64}, Float64, Vec{3,Float64}}
    end

    @testset "Vector field (Displacement)" begin
        # Create displacement field at a point
        u = Vec{3}((0.1, 0.2, 0.3))
        ∇u = Tensor{2,3}((1.0, 0.1, 0.0,
                          0.1, 1.0, 0.0,
                          0.0, 0.0, 1.0))
        u̇ = Vec{3}((0.01, 0.02, 0.03))
        ∇u̇ = Tensor{2,3}((0.001, 0.0, 0.0,
                          0.0, 0.001, 0.0,
                          0.0, 0.0, 0.001))

        u_local = LocalField(u, ∇u, u̇, ∇u̇)

        @test u_local.value == u
        @test u_local.gradient == ∇u
        @test u_local.rate == u̇
        @test u_local.gradient_rate == ∇u̇

        # Type stability
        @test typeof(u_local) == LocalField{Vec{3,Float64}, Tensor{2,3,Float64,9}, Vec{3,Float64}, Tensor{2,3,Float64,9}}
    end

    @testset "Quasi-static case (rate = 0)" begin
        # Quasi-static: rate = 0, but gradient_rate is computed from increments
        u = Vec{3}((0.1, 0.2, 0.3))
        ∇u = Tensor{2,3}((1.01, 0.0, 0.0,
                          0.0, 1.02, 0.0,
                          0.0, 0.0, 1.03))
        u̇ = zero(Vec{3})  # No velocity in quasi-static
        ∇u̇ = Tensor{2,3}((0.01, 0.0, 0.0,
                          0.0, 0.02, 0.0,
                          0.0, 0.0, 0.03))  # Still computed from increment!

        u_local = LocalField(u, ∇u, u̇, ∇u̇)

        @test u_local.value == u
        @test u_local.rate == zero(Vec{3})
        @test u_local.gradient_rate ≠ zero(Tensor{2,3})  # Gradient rate is NOT zero!
    end

    # NOTE: create_local_field_namedtuple() was removed - use interpolate_local_fields() instead
    # See test/elements/test_interpolate_local_fields.jl for the real implementation tests
end
