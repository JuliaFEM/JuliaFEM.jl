# Test that basis functions return SVector types
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
