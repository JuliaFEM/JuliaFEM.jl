# Test basis function derivatives
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
