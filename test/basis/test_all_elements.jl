# Comprehensive tests for all 14 basis families
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

@testset "All Element Types - Partition of Unity" begin
    # 1D elements
    @testset "Seg2" begin
        N = get_basis_functions(Seg2(), Lagrange{1}(), Vec{1}((0.0,)))
        @test sum(N) ≈ 1.0 atol = 1e-14
        @test length(N) == 2
    end

    @testset "Seg3" begin
        N = get_basis_functions(Seg3(), Lagrange{2}(), Vec{1}((0.5,)))
        @test sum(N) ≈ 1.0 atol = 1e-14
        @test length(N) == 3
    end

    # 2D triangular elements
    @testset "Tri3" begin
        N = get_basis_functions(Tri3(), Lagrange{1}(), Vec{2}((0.3, 0.3)))
        @test sum(N) ≈ 1.0 atol = 1e-14
        @test length(N) == 3
    end

    @testset "Tri6" begin
        N = get_basis_functions(Tri6(), Lagrange{2}(), Vec{2}((0.4, 0.2)))
        @test sum(N) ≈ 1.0 atol = 1e-14
        @test length(N) == 6
    end

    # 2D quadrilateral elements
    @testset "Quad4" begin
        N = get_basis_functions(Quad4(), Lagrange{1}(), Vec{2}((0.0, 0.0)))
        @test sum(N) ≈ 1.0 atol = 1e-14
        @test length(N) == 4
    end

    @testset "Quad8" begin
        N = get_basis_functions(Quad8(), Serendipity{2}(), Vec{2}((-0.5, 0.5)))
        @test sum(N) ≈ 1.0 atol=1e-14
        @test length(N) == 8
    end

    @testset "Quad9" begin
        N = get_basis_functions(Quad9(), Lagrange{2}(), Vec{2}((0.0, 0.0)))
        @test sum(N) ≈ 1.0 atol = 1e-14
        @test length(N) == 9
    end

    # 3D tetrahedral elements
    @testset "Tet4" begin
        N = get_basis_functions(Tet4(), Lagrange{1}(), Vec{3}((0.25, 0.25, 0.25)))
        @test sum(N) ≈ 1.0 atol = 1e-14
        @test length(N) == 4
    end

    @testset "Tet10" begin
        N = get_basis_functions(Tet10(), Lagrange{2}(), Vec{3}((0.2, 0.3, 0.1)))
        @test sum(N) ≈ 1.0 atol = 1e-14
        @test length(N) == 10
    end

    # 3D hexahedral elements
    @testset "Hex8" begin
        N = get_basis_functions(Hex8(), Lagrange{1}(), Vec{3}((0.0, 0.0, 0.0)))
        @test sum(N) ≈ 1.0 atol = 1e-14
        @test length(N) == 8
    end

    @testset "Hex20" begin
        N = get_basis_functions(Hex20(), Serendipity{2}(), Vec{3}((0.5, -0.5, 0.0)))
        @test sum(N) ≈ 1.0 atol=1e-14
        @test length(N) == 20
    end

    @testset "Hex27" begin
        N = get_basis_functions(Hex27(), Lagrange{2}(), Vec{3}((0.0, 0.0, 0.0)))
        @test sum(N) ≈ 1.0 atol = 1e-14
        @test length(N) == 27
    end

    # 3D pyramid element
    @testset "Pyr5" begin
        N = get_basis_functions(Pyr5(), Lagrange{1}(), Vec{3}((0.0, 0.0, 0.2)))
        @test sum(N) ≈ 1.0 atol = 1e-14
        @test length(N) == 5
    end

    # 3D wedge element
    @testset "Wedge6" begin
        N = get_basis_functions(Wedge6(), Lagrange{1}(), Vec{3}((0.2, 0.3, 0.0)))
        @test sum(N) ≈ 1.0 atol = 1e-14
        @test length(N) == 6
    end
end
