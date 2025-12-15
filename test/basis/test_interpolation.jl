# Test interpolation with basis functions
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
