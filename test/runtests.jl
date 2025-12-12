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
        @test LocalField isa Type
    end

    # Field tests
    @testset "Fields" begin
        include("fields/test_local_field.jl")
    end

    # Physics tests
    @testset "Physics" begin
        include("physics/test_strain.jl")
    end

    # Element interpolation tests
    @testset "Element Interpolation" begin
        include("elements/test_interpolate_local_fields.jl")
    end

    # Material tests
    @testset "Materials" begin
        include("materials/test_state_variables.jl")
        include("materials/test_traits.jl")
        include("materials/test_global_material_cache.jl")
        include("materials/test_plasticity_integration.jl")
        include("materials/test_cantilever_material_cache_zero_allocations.jl")
        include("materials/test_assembly_workspace_refactor.jl")
    end

    # Topology tests
    @testset "Topology" begin
        include("topology/test_segments.jl")
        include("topology/test_triangles.jl")
        include("topology/test_quadrilaterals.jl")
        include("topology/test_topology_entities_tetrahedron.jl")
        include("topology/test_hexahedra.jl")
        include("topology/test_pyramids.jl")
        include("topology/test_wedges.jl")
        # include("topology/test_topology_type_extraction.jl")  # DISABLED: obsolete API (entities don't carry topology)
        include("topology/test_topological_invariance.jl")
        include("topology/test_entity_dimensions.jl")
    end

    # Domain-specific tests
    @testset "Continuum Domain" begin
        include("domains/continuum/runtests.jl")
    end

    # Validation tests
    include("validation/test_cantilever_regression.jl")
    include("validation/test_plasticity_simple.jl")
end

# Note: All legacy tests have been moved to test/broken/
# They need to be updated to work with the new modular architecture.
# To run a specific test, use:
#   include("test/broken/test_name.jl")
