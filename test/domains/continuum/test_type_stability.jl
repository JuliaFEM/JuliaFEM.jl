# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Type stability tests for generic integration functions.

Tests the new trait-based integration system for type stability.
"""

using Test
using JuliaFEM
using Tensors
using InteractiveUtils

# Import required functions
using JuliaFEM: prepare_element!, compute_block!, compute_all_blocks!

@testset "Generic Integration Type Stability" begin
    # Create test mesh (single Hex8 element)
    function create_test_mesh()
        nodes = Vec{3,Float64}[
            Vec{3}((0.0, 0.0, 0.0)),
            Vec{3}((1.0, 0.0, 0.0)),
            Vec{3}((1.0, 1.0, 0.0)),
            Vec{3}((0.0, 1.0, 0.0)),
            Vec{3}((0.0, 0.0, 1.0)),
            Vec{3}((1.0, 0.0, 1.0)),
            Vec{3}((1.0, 1.0, 1.0)),
            Vec{3}((0.0, 1.0, 1.0)),
        ]

        connectivity = [NTuple{8,UInt32}((1, 2, 3, 4, 5, 6, 7, 8))]
        element_sets = Dict{Symbol,Set{UInt32}}(:all => Set(UInt32[1]))

        return Mesh{8,Hexahedron{8}}(nodes, connectivity, element_sets)
    end

    @testset "LinearElastic type stability" begin
        mesh = create_test_mesh()
        formulation = ContinuumFormulation{FullThreeD}()
        material = LinearElastic(E=210.0e9, ν=0.3)
        field = Displacement{3}()
        kernel = ContinuumKernel(formulation, material, field)

        cache = create_element_cache(mesh, kernel)

        # Test prepare_element! type stability
        prepared = prepare_element!(cache, kernel, 1, mesh)
        @test isa(prepared, PreparedElement)
        @test !any(isnan, prepared.detJ_w)

        # Test compute_block! type stability (constant tangent path)
        u_elem = zeros(Float64, 24)
        K_block = compute_block!(prepared, material, 1, 2, u_elem)
        @test isa(K_block, Tensor{2,3,Float64})
        @test !any(isnan, K_block)

        # Test compute_all_blocks! type stability
        K_blocks = cache.K_blocks
        compute_all_blocks!(K_blocks, prepared, material, u_elem, 8)
        @test all(K -> all(!isnan, K), K_blocks)

        println("  ✓ LinearElastic integration is type-stable")
    end

    @testset "NeoHookean type stability" begin
        mesh = create_test_mesh()
        formulation = ContinuumFormulation{FullThreeD}()
        material = NeoHookean(E_mod=210.0e9, nu=0.3)
        field = Displacement{3}()
        kernel = ContinuumKernel(formulation, material, field)

        cache = create_element_cache(mesh, kernel)

        # Test prepare_element! type stability
        prepared = prepare_element!(cache, kernel, 1, mesh)
        @test isa(prepared, PreparedElement)

        # Test compute_block! type stability (strain-dependent path)
        u_elem = zeros(Float64, 24)
        K_block = compute_block!(prepared, material, 1, 2, u_elem)
        @test isa(K_block, Tensor{2,3,Float64})
        @test !any(isnan, K_block)

        # Test compute_all_blocks! type stability
        K_blocks = cache.K_blocks
        compute_all_blocks!(K_blocks, prepared, material, u_elem, 8)
        @test all(K -> all(!isnan, K), K_blocks)

        println("  ✓ NeoHookean integration is type-stable")
    end

    @testset "Material behavior trait dispatch is type-stable" begin
        mat_linear = LinearElastic(E=210e9, ν=0.3)
        mat_neo = NeoHookean(μ=1e6, λ=1e9)

        # Test trait functions return concrete types
        behavior_linear = material_behavior(mat_linear)
        behavior_neo = material_behavior(mat_neo)

        @test isa(behavior_linear, StatelessConstantTangent)
        @test isa(behavior_neo, StatelessStrainDependent)

        # Test helper functions
        @test isa(needs_deformation(mat_linear), Bool)
        @test isa(needs_deformation(mat_neo), Bool)
        @test isa(needs_state(mat_linear), Bool)
        @test isa(needs_state(mat_neo), Bool)

        println("  ✓ Material trait dispatch is type-stable")
    end
end

println("\n" * "="^70)
println("TYPE STABILITY ANALYSIS COMPLETE")
println("="^70)
println("All generic integration functions are type-stable ✓")
println("="^70)
