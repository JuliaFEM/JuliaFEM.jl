# Test type stability and type parameters
# Generated functions should work with Float32, Float64, BigFloat, and dual numbers
using Test
using StaticArrays, Tensors

include("../../src/topology/api.jl")
include("../../src/topology/segments.jl")
include("../../src/topology/triangles.jl")
include("../../src/topology/quadrilaterals.jl")
include("../../src/topology/tetrahedra.jl")
include("../../src/topology/hexahedra.jl")
include("../../src/topology/pyramids.jl")
include("../../src/topology/wedges.jl")
include("../../src/basis/api.jl")
include("../../src/basis/basis_generated.jl")

@testset "Type Stability" begin
    @testset "Float64" begin
        xi = Vec{2,Float64}((0.5, 0.3))
        N = get_basis_functions(Tri3(), Lagrange{1}(), xi)
        @test N isa SVector{3,Float64}
        @test eltype(N) === Float64

        dN = get_basis_derivatives(Tri3(), Lagrange{1}(), xi)
        @test dN isa SVector{3,Vec{2,Float64}}
        @test eltype(dN) === Vec{2,Float64}
    end

    @testset "Float32" begin
        xi = Vec{2,Float32}((0.5f0, 0.3f0))
        N = get_basis_functions(Tri3(), Lagrange{1}(), xi)
        @test N isa SVector{3,Float32}
        @test eltype(N) === Float32

        dN = get_basis_derivatives(Tri3(), Lagrange{1}(), xi)
        @test dN isa SVector{3,Vec{2,Float32}}
        @test eltype(dN) === Vec{2,Float32}
    end

    @testset "BigFloat" begin
        xi = Vec{2,BigFloat}((BigFloat("0.5"), BigFloat("0.3")))
        N = get_basis_functions(Tri3(), Lagrange{1}(), xi)
        @test N isa SVector{3,BigFloat}
        @test eltype(N) === BigFloat

        dN = get_basis_derivatives(Tri3(), Lagrange{1}(), xi)
        @test dN isa SVector{3,Vec{2,BigFloat}}
        @test eltype(dN) === Vec{2,BigFloat}
    end

    @testset "3D elements with different types" begin
        # Test Hex8 with Float32
        xi32 = Vec{3,Float32}((0.0f0, 0.0f0, 0.0f0))
        N32 = get_basis_functions(Hex8(), Lagrange{1}(), xi32)
        @test N32 isa SVector{8,Float32}

        # Test Tet4 with Float64
        xi64 = Vec{3,Float64}((0.25, 0.25, 0.25))
        N64 = get_basis_functions(Tet4(), Lagrange{1}(), xi64)
        @test N64 isa SVector{4,Float64}
    end
end
