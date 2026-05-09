"""
Test suite for continuum mechanics domain.

Organized by priority:
1. Fix errors first
2. Then correctness
3. Finally zero allocations

Run with: julia --project=. test/domains/continuum/runtests.jl
"""

using Test
using JuliaFEM
using Tensors
using LinearAlgebra

# Load shared helper functions
include("test_helpers.jl")

@testset "Continuum Domain Tests" begin

    @testset "Material Trait System" begin
        @testset "LinearElastic traits" begin
            mat = LinearElastic(E=210e9, ν=0.3)
            @test material_behavior(mat) isa StatelessConstantTangent
            @test needs_deformation(mat) == false
            @test needs_state(mat) == false
        end

        @testset "NeoHookean traits" begin
            mat = NeoHookean(μ=1e6, λ=1e9)
            @test material_behavior(mat) isa StatelessStrainDependent
            @test needs_deformation(mat) == true
            @test needs_state(mat) == false
        end
    end

    @testset "Hex8 Element Validation (Correctness)" begin
        include("test_validation_hex8.jl")
    end

    @testset "DOF Mapping (Correctness)" begin
        include("test_dofs_per_node.jl")
        include("test_dof_mapping.jl")
    end

    @testset "Cache Reset Functions" begin
        include("test_reset_functions.jl")
    end

    @testset "Cache Update Functions (Three-Phase API)" begin
        include("test_cache_updates.jl")
        include("test_material_cache_branches.jl")
    end

    @testset "Kernel Functions" begin
        include("test_kernel_functions.jl")
        include("test_compute_block.jl")
    end

    @testset "Full Assembly Workflow" begin
        include("test_full_assembly.jl")
    end

    @testset "Continuum brick direct solve" begin
        include("test_continuum_brick_solve.jl")
    end

    @testset "Material element lab (single Hex8 coupon)" begin
        include("test_material_element_lab.jl")
    end

    @testset "Mixed u–p (MixedUPKernel)" begin
        include("test_mixed_up_kernel.jl")
        include("test_mixed_up_incompressible_solve.jl")
        include("test_approx_schur_diag_preconditioner.jl")
    end

    @testset "Stokes mixed (StokesMixedKernel)" begin
        include("test_stokes_mixed_kernel.jl")
        include("test_stokes_mixed_incompressible_solve.jl")
    end

    @testset "Hellinger–Reissner (HellingerReissnerKernel)" begin
        include("test_hellinger_reissner_kernel.jl")
        include("test_hellinger_reissner_matvec_penalty.jl")
    end

    @testset "Hu–Washizu (HuWashizuKernel)" begin
        include("test_hu_washizu_kernel.jl")
        include("test_hu_washizu_matvec_penalty.jl")
    end
end

println("\n" * "="^70)
println("CONTINUUM DOMAIN TEST SUMMARY")
println("="^70)
println("Test organization:")
println("  1. Material traits - Type system verification")
println("  2. Hex8 validation - Correctness against analytical solution")
println("  3. DOF mapping - Index computation correctness")
println("  4. Cache resets - Zero-allocation reset functions")
println("  5. Cache updates - Three-phase update workflow")
println("  6. Kernel functions - Weak form and integration")
println("  7. Full assembly - Complete workflow validation")
println("="^70)
