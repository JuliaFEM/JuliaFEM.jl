# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using Test
using JuliaFEM
using Tensors

uint_tuple(n::Int) = tuple([UInt(i) for i in 1:n]...)

# Regression coverage for the interpolate_field_value vector branch.
# Previously the @generated body referenced the function `quantity_type`
# instead of the local variable `Q`, so the elseif silently failed and
# the vector branch was unreachable. These tests would have errored out
# before the fix.
@testset "interpolate_field_value" begin
    @testset "Scalar (Temperature) field" begin
        S = @DOFSet{T::DOF{Temperature, Vertex}}
        elem = Element{Tetrahedron{4}, Lagrange{1}, S, 4}(UInt(1), uint_tuple(4))

        # Use a constant field; any interpolation must reproduce that value.
        u_global = fill(2.5, 4)
        ξ = Vec((0.25, 0.25, 0.25))

        T_val = interpolate_field_value(elem, u_global, :T, ξ)
        @test T_val isa Float64
        @test T_val ≈ 2.5

        @inferred interpolate_field_value(elem, u_global, :T, ξ)
    end

    @testset "Vector (Displacement{3}) field" begin
        S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
        elem = Element{Tetrahedron{4}, Lagrange{1}, S, 12}(UInt(1), uint_tuple(12))

        # Constant per-component values; vector interpolation must reproduce
        # the per-component constant.
        u_global = Float64[
            1.0, 2.0, 3.0,  # Node 1
            1.0, 2.0, 3.0,  # Node 2
            1.0, 2.0, 3.0,  # Node 3
            1.0, 2.0, 3.0,  # Node 4
        ]
        ξ = Vec((0.25, 0.25, 0.25))

        u_val = interpolate_field_value(elem, u_global, :u, ξ)
        @test u_val isa Vec{3, Float64}
        @test u_val ≈ Vec{3}((1.0, 2.0, 3.0))

        @inferred interpolate_field_value(elem, u_global, :u, ξ)
    end

    @testset "Multi-field (T + u)" begin
        S = @DOFSet{T::DOF{Temperature, Vertex}, u::DOF{Displacement{3}, Vertex}}
        elem = Element{Tetrahedron{4}, Lagrange{1}, S, 16}(UInt(1), uint_tuple(16))

        # First 4 entries are temperature, next 12 are displacement.
        u_global = Float64[
            5.0, 5.0, 5.0, 5.0,
            0.5, 0.0, 0.0,
            0.5, 0.0, 0.0,
            0.5, 0.0, 0.0,
            0.5, 0.0, 0.0,
        ]
        ξ = Vec((0.25, 0.25, 0.25))

        @test interpolate_field_value(elem, u_global, :T, ξ) ≈ 5.0
        @test interpolate_field_value(elem, u_global, :u, ξ) ≈ Vec{3}((0.5, 0.0, 0.0))

        @test_throws ErrorException interpolate_field_value(elem, u_global, :nope, ξ)
    end
end
