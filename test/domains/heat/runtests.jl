"""
Test suite for the heat conduction domain.

Heat is the second physics domain to ride the new microkernel contract
(`reference_fields`, `qpoint_buffer_eltype`, `update_qpoint_buffer!`,
`evaluate_entry`). The tests here are deliberately structured to mirror
`test/domains/continuum/runtests.jl`, so any drift between the two
domains is visible at a glance.

Run with: julia --project=. test/domains/heat/runtests.jl
"""

using Test
using JuliaFEM
using Tensors
using LinearAlgebra

@testset "Heat Domain Tests" begin

    @testset "Material Trait System" begin
        @testset "HeatConductivity traits" begin
            mat = HeatConductivity(k = 50.2)
            @test material_behavior(mat) isa StatelessConstantTangent
            @test needs_state(mat) == false
        end
    end

    @testset "Kernel through DOF-based assembler" begin
        # Correctness, KA equivalence, zero-alloc, matrix-free CG, type
        # stability — single consolidated file so any regression in the
        # assembler that breaks heat fails on a clearly named test.
        include("test_heat_kernel.jl")
    end
end

println("\n" * "="^70)
println("HEAT DOMAIN TEST SUMMARY")
println("="^70)
println("Test organization:")
println("  1. Material traits  - Type system verification")
println("  2. Kernel through DOF-based assembler:")
println("     - assemble! correctness (symmetry, null space, K vs apply_K!)")
println("     - KA apply_K! (CPU backend) bit-equivalence")
println("     - Zero allocation + 0 LLVM gc-alloc sites")
println("     - Matrix-free CG via LinearOperators (penalty Dirichlet)")
println("     - Type stability of assemble! + apply_K!")
println("="^70)
