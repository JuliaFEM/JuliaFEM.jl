# Test partition of unity property: sum(N) = 1
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
