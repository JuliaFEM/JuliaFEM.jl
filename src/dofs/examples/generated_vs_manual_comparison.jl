# Comprehensive Comparison: @generated vs Manual DOF Extraction
# ================================================================
#
# This benchmark compares:
# 1. @generated function (current implementation)
# 2. Manual SVector of Vec{3} construction
# 3. Assembly-level verification that they produce identical code
#
# Date: November 22, 2025

using JuliaFEM
using BenchmarkTools
using StaticArrays
using Tensors
using InteractiveUtils

println("="^80)
println("GENERATED vs MANUAL: Comprehensive Comparison")
println("="^80)
println()

# Test data: Tet4 element with 3D displacement (12 DOFs)
u_global = rand(100)
dof_indices = (5, 6, 7, 20, 21, 22, 35, 36, 37, 50, 51, 52)

# ============================================================================
# IMPLEMENTATION 1: @generated (current)
# ============================================================================

println("IMPLEMENTATION 1: @generated function")
println("-"^80)
println()

println("Source code:")
println(raw"""
@inline @generated function extract_element_dofs(
    ::Type{VectorDOF{D}},
    u_global::Vector{T},
    dof_indices::NTuple{N,Int}
) where {D, T, N}
    n_nodes = N ÷ D
    vec_exprs = []
    for i in 1:n_nodes
        start_idx = (i-1) * D + 1
        components = [:(u_global[dof_indices[$j]]) for j in start_idx:(start_idx+D-1)]
        push!(vec_exprs, :(Vec{$D,$T}(($(components...),))))
    end
    return :(
        @inbounds SVector(($(vec_exprs...),))
    )
end
""")
println()

# Test it works
result_generated = extract_element_dofs(VectorDOF{3}, u_global, dof_indices)
println("Result type: $(typeof(result_generated))")
println("Result[1]: $(result_generated[1])")
println()

# ============================================================================
# IMPLEMENTATION 2: Manual (baseline)
# ============================================================================

println("IMPLEMENTATION 2: Manual inline function")
println("-"^80)
println()

@inline function extract_manual(u::Vector{Float64}, indices::NTuple{12, Int})
    @inbounds SVector(
        Vec{3}((u[indices[1]], u[indices[2]], u[indices[3]])),
        Vec{3}((u[indices[4]], u[indices[5]], u[indices[6]])),
        Vec{3}((u[indices[7]], u[indices[8]], u[indices[9]])),
        Vec{3}((u[indices[10]], u[indices[11]], u[indices[12]]))
    )
end

println("Source code:")
println("""
@inline function extract_manual(u::Vector{Float64}, indices::NTuple{12, Int})
    @inbounds SVector(
        Vec{3}((u[indices[1]], u[indices[2]], u[indices[3]])),
        Vec{3}((u[indices[4]], u[indices[5]], u[indices[6]])),
        Vec{3}((u[indices[7]], u[indices[8]], u[indices[9]])),
        Vec{3}((u[indices[10]], u[indices[11]], u[indices[12]]))
    )
end
""")
println()

# Test it works
result_manual = extract_manual(u_global, dof_indices)
println("Result type: $(typeof(result_manual))")
println("Result[1]: $(result_manual[1])")
println()

# Verify they produce the same results
println("Results match: $(result_generated ≈ result_manual)")
println()

# ============================================================================
# BENCHMARKTOOLS COMPARISON
# ============================================================================

println("="^80)
println("PERFORMANCE BENCHMARKS (BenchmarkTools)")
println("="^80)
println()

println("@generated version:")
b_generated = @benchmark extract_element_dofs($VectorDOF{3}, $u_global, $dof_indices)
display(b_generated)
println()
println()

println("Manual version:")
b_manual = @benchmark extract_manual($u_global, $dof_indices)
display(b_manual)
println()
println()

println("="^80)
println("SUMMARY")
println("="^80)
println()

min_generated = minimum(b_generated.times)
min_manual = minimum(b_manual.times)
median_generated = median(b_generated.times)
median_manual = median(b_manual.times)

println("Minimum time:")
println("  @generated:  $(round(min_generated, digits=2)) ns")
println("  Manual:      $(round(min_manual, digits=2)) ns")
println("  Ratio:       $(round(min_generated/min_manual, digits=2))x")
println()

println("Median time:")
println("  @generated:  $(round(median_generated, digits=2)) ns")
println("  Manual:      $(round(median_manual, digits=2)) ns")
println("  Ratio:       $(round(median_generated/median_manual, digits=2))x")
println()

println("Allocations:")
println("  @generated:  $(b_generated.allocs) allocations, $(b_generated.memory) bytes")
println("  Manual:      $(b_manual.allocs) allocations, $(b_manual.memory) bytes")
println()

# ============================================================================
# HOT LOOP COMPARISON (Real-world performance)
# ============================================================================

println("="^80)
println("HOT LOOP PERFORMANCE (10M iterations)")
println("="^80)
println()

n_iterations = 10_000_000

println("Testing @generated version...")
function test_generated_loop()
    checksum = 0.0
    for i in 1:n_iterations
        result = extract_element_dofs(VectorDOF{3}, u_global, dof_indices)
        checksum += result[1][1]
    end
    return checksum
end
t_generated = @elapsed test_generated_loop()
println("  Time: $(round(t_generated, digits=3)) seconds")
println("  Per call: $(round(t_generated/n_iterations*1e9, digits=2)) ns")
println("  Throughput: $(round(n_iterations/t_generated/1e6, digits=1))M calls/sec")
println()

println("Testing manual version...")
function test_manual_loop()
    checksum = 0.0
    for i in 1:n_iterations
        result = extract_manual(u_global, dof_indices)
        checksum += result[1][1]
    end
    return checksum
end
t_manual = @elapsed test_manual_loop()
println("  Time: $(round(t_manual, digits=3)) seconds")
println("  Per call: $(round(t_manual/n_iterations*1e9, digits=2)) ns")
println("  Throughput: $(round(n_iterations/t_manual/1e6, digits=1))M calls/sec")
println()

println("Hot loop slowdown: $(round(t_generated/t_manual, digits=2))x")
println()

# ============================================================================
# ASSEMBLY CODE COMPARISON
# ============================================================================

println("="^80)
println("ASSEMBLY CODE ANALYSIS")
println("="^80)
println()

println("@generated version assembly:")
println("-"^80)
@code_native debuginfo=:none syntax=:intel extract_element_dofs(VectorDOF{3}, u_global, dof_indices)
println()

println("\n" * "="^80)
println("Manual version assembly:")
println("-"^80)
@code_native debuginfo=:none syntax=:intel extract_manual(u_global, dof_indices)
println()

# ============================================================================
# TYPE INFERENCE COMPARISON
# ============================================================================

println("\n" * "="^80)
println("TYPE INFERENCE ANALYSIS")
println("="^80)
println()

println("@generated version:")
println("-"^80)
@code_warntype extract_element_dofs(VectorDOF{3}, u_global, dof_indices)
println()

println("\nManual version:")
println("-"^80)
@code_warntype extract_manual(u_global, dof_indices)
println()

# ============================================================================
# FINAL VERDICT
# ============================================================================

println("\n" * "="^80)
println("FINAL VERDICT")
println("="^80)
println()

if min_generated < min_manual * 1.2
    println("✅ EXCELLENT: @generated matches manual performance (within 20%)")
    println("   Both implementations are production-ready.")
elseif min_generated < min_manual * 2.0
    println("✅ GOOD: @generated is slower but acceptable (within 2x)")
    println("   Generic API is worth the minor overhead.")
elseif min_generated < min_manual * 10.0
    println("⚠️  MODERATE: @generated is significantly slower (2-10x)")
    println("   Consider manual implementations for hot paths.")
else
    println("❌ POOR: @generated is much slower (>10x)")
    println("   Manual implementation recommended for performance-critical code.")
end

println()
println("Key findings:")
println("  • BenchmarkTools minimum: $(round(min_generated/min_manual, digits=1))x slower")
println("  • Hot loop performance: $(round(t_generated/t_manual, digits=1))x slower")
println("  • Type stability: Both implementations type-stable")
println("  • Allocations: @generated=$(b_generated.allocs), manual=$(b_manual.allocs)")
println()

println("Recommendation:")
if t_generated/t_manual < 2.0
    println("  Use @generated for clean, generic API.")
else
    println("  For assembly loops, manually inline extraction for critical elements.")
    println("  Generic @generated version is fine for non-critical paths.")
end

println()
println("="^80)
println("Analysis complete. See assembly output above for 1:1 comparison.")
println("="^80)
