# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM, Test

@testset "JuliaFEM.jl" begin
    @testset "Package loads" begin
        @test isdefined(JuliaFEM, :AbstractBasis)
        @test isdefined(JuliaFEM, :AbstractMesh)
        @test isdefined(JuliaFEM, :AbstractTopology)
        @test isdefined(JuliaFEM, :AbstractMaterial)
        @test isdefined(JuliaFEM, :AbstractPhysics)
        @test isdefined(JuliaFEM, :AbstractField)
        @test isdefined(JuliaFEM, :AbstractFormulation)
    end

    @testset "Basic types instantiate" begin
        # Test that basic types can be referenced
        @test AbstractBasis isa Type
        @test AbstractMesh isa Type
        @test AbstractTopology isa Type
        @test AbstractMaterial isa Type
        @test AbstractPhysics isa Type
        @test AbstractField isa Type
        @test AbstractFormulation isa Type
    end

    @testset "Concrete types exist" begin
        # Test that concrete implementations exist
        @test Mesh isa Type
        @test LinearElastic isa Type
        @test Hex8 isa Type
        @test Tet4 isa Type
    end

    # Domain-specific tests
    @testset "Continuum Domain" begin
        include("domains/continuum/runtests.jl")
    end

    # Validation tests
    include("validation/test_cantilever_regression.jl")
end

# Note: All legacy tests have been moved to test/broken/
# They need to be updated to work with the new modular architecture.
# To run a specific test, use:
#   include("test/broken/test_name.jl")
