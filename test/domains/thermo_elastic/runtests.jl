"""
Test suite for the thermo-elastic domain.

`ThermoElasticKernel` is the **first multi-field** kernel in the
codebase. The tests here exercise the multi-field path through:

  - `local_dof_layout(E)` (mixed `field_idx` for `u` + `T`),
  - `_prepare_caches!` (must read DOFs from `elem.dof_indices`),
  - `evaluate_entry` ((field_i, field_j) dispatch into K_uu / K_TT /
    K_uT / K_Tu blocks),
  - the matrix-free `apply_K!` (CPU + KernelAbstractions CPU backend).

If anything in the assembler ever silently re-mono-field-ifies (e.g.
hardcodes `dofs_per_node(get_field(kernel))` somewhere new), this file
fails before any user does.

Run with: julia --project=. test/domains/thermo_elastic/runtests.jl
"""

using Test
using JuliaFEM
using Tensors
using LinearAlgebra

@testset "Thermo-Elastic Domain Tests" begin
    @testset "Multi-field DOF assembly through DOF-based assembler" begin
        include("test_thermo_elastic_kernel.jl")
    end
end

println("\n" * "="^70)
println("THERMO-ELASTIC DOMAIN TEST SUMMARY")
println("="^70)
println("Test organization (multi-field validation):")
println("  1. local_dof_layout(E) for (u + T) returns 32 entries with mixed field_idx")
println("  2. β = 0  ⇒  K_uu and K_TT exactly match standalone elasticity / heat")
println("  3. β ≠ 0  ⇒  off-diagonal K_uT / K_Tu non-zero, K symmetric,")
println("              apply_K! (CPU + KA) matches K * x")
println("  4. assemble! and apply_K! both zero-allocation in the hot loop")
println("="^70)
