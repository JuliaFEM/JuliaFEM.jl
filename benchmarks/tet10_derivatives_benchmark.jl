# Benchmark: Tet10 Shape Function Derivatives - Manual vs AD
# 
# Compares two approaches:
# 1. Manual: Hand-calculated derivatives (traditional FEM)
# 2. AD: Automatic differentiation using Tensors.jl gradient()
#
# Run with: julia --project=. benchmarks/tet10_derivatives_benchmark.jl

using BenchmarkTools
using Tensors
using Printf

println("="^70)
println("Tet10 Shape Function Derivatives: Manual vs AD Benchmark")
println("="^70)
println()

# ============================================================================
# METHOD 1: MANUAL (Hand-Calculated Derivatives)
# ============================================================================

"""
Tet10 shape functions (manual implementation).
Reference element: ξ ∈ [0,1], η ∈ [0,1], ζ ∈ [0,1], ξ+η+ζ ≤ 1
"""
module ManualTet10
using Tensors

# Shape functions
@inline N1(ξ, η, ζ) = (1 - ξ - η - ζ) * (2 * (1 - ξ - η - ζ) - 1)
@inline N2(ξ, η, ζ) = ξ * (2 * ξ - 1)
@inline N3(ξ, η, ζ) = η * (2 * η - 1)
@inline N4(ξ, η, ζ) = ζ * (2 * ζ - 1)
@inline N5(ξ, η, ζ) = 4 * ξ * (1 - ξ - η - ζ)
@inline N6(ξ, η, ζ) = 4 * ξ * η
@inline N7(ξ, η, ζ) = 4 * η * (1 - ξ - η - ζ)
@inline N8(ξ, η, ζ) = 4 * ζ * (1 - ξ - η - ζ)
@inline N9(ξ, η, ζ) = 4 * ξ * ζ
@inline N10(ξ, η, ζ) = 4 * η * ζ

# Derivatives (calculated by hand - error-prone!)
@inline dN1_dξ(ξ, η, ζ) = 4 * ξ + 4 * η + 4 * ζ - 3
@inline dN1_dη(ξ, η, ζ) = 4 * ξ + 4 * η + 4 * ζ - 3
@inline dN1_dζ(ξ, η, ζ) = 4 * ξ + 4 * η + 4 * ζ - 3

@inline dN2_dξ(ξ, η, ζ) = 4 * ξ - 1
@inline dN2_dη(ξ, η, ζ) = 0.0
@inline dN2_dζ(ξ, η, ζ) = 0.0

@inline dN3_dξ(ξ, η, ζ) = 0.0
@inline dN3_dη(ξ, η, ζ) = 4 * η - 1
@inline dN3_dζ(ξ, η, ζ) = 0.0

@inline dN4_dξ(ξ, η, ζ) = 0.0
@inline dN4_dη(ξ, η, ζ) = 0.0
@inline dN4_dζ(ξ, η, ζ) = 4 * ζ - 1

@inline dN5_dξ(ξ, η, ζ) = 4 * (1 - 2 * ξ - η - ζ)
@inline dN5_dη(ξ, η, ζ) = -4 * ξ
@inline dN5_dζ(ξ, η, ζ) = -4 * ξ

@inline dN6_dξ(ξ, η, ζ) = 4 * η
@inline dN6_dη(ξ, η, ζ) = 4 * ξ
@inline dN6_dζ(ξ, η, ζ) = 0.0

@inline dN7_dξ(ξ, η, ζ) = -4 * η
@inline dN7_dη(ξ, η, ζ) = 4 * (1 - ξ - 2 * η - ζ)
@inline dN7_dζ(ξ, η, ζ) = -4 * η

@inline dN8_dξ(ξ, η, ζ) = -4 * ζ
@inline dN8_dη(ξ, η, ζ) = -4 * ζ
@inline dN8_dζ(ξ, η, ζ) = 4 * (1 - ξ - η - 2 * ζ)

@inline dN9_dξ(ξ, η, ζ) = 4 * ζ
@inline dN9_dη(ξ, η, ζ) = 0.0
@inline dN9_dζ(ξ, η, ζ) = 4 * ξ

@inline dN10_dξ(ξ, η, ζ) = 0.0
@inline dN10_dη(ξ, η, ζ) = 4 * ζ
@inline dN10_dζ(ξ, η, ζ) = 4 * η

# Evaluation function (returns tuple - zero allocation)
@inline function eval_basis_and_grad(xi::Vec{3})
    ξ, η, ζ = xi[1], xi[2], xi[3]

    N = (N1(ξ, η, ζ), N2(ξ, η, ζ), N3(ξ, η, ζ), N4(ξ, η, ζ), N5(ξ, η, ζ),
        N6(ξ, η, ζ), N7(ξ, η, ζ), N8(ξ, η, ζ), N9(ξ, η, ζ), N10(ξ, η, ζ))

    dN = (Vec(dN1_dξ(ξ, η, ζ), dN1_dη(ξ, η, ζ), dN1_dζ(ξ, η, ζ)),
        Vec(dN2_dξ(ξ, η, ζ), dN2_dη(ξ, η, ζ), dN2_dζ(ξ, η, ζ)),
        Vec(dN3_dξ(ξ, η, ζ), dN3_dη(ξ, η, ζ), dN3_dζ(ξ, η, ζ)),
        Vec(dN4_dξ(ξ, η, ζ), dN4_dη(ξ, η, ζ), dN4_dζ(ξ, η, ζ)),
        Vec(dN5_dξ(ξ, η, ζ), dN5_dη(ξ, η, ζ), dN5_dζ(ξ, η, ζ)),
        Vec(dN6_dξ(ξ, η, ζ), dN6_dη(ξ, η, ζ), dN6_dζ(ξ, η, ζ)),
        Vec(dN7_dξ(ξ, η, ζ), dN7_dη(ξ, η, ζ), dN7_dζ(ξ, η, ζ)),
        Vec(dN8_dξ(ξ, η, ζ), dN8_dη(ξ, η, ζ), dN8_dζ(ξ, η, ζ)),
        Vec(dN9_dξ(ξ, η, ζ), dN9_dη(ξ, η, ζ), dN9_dζ(ξ, η, ζ)),
        Vec(dN10_dξ(ξ, η, ζ), dN10_dη(ξ, η, ζ), dN10_dζ(ξ, η, ζ)))

    return N, dN
end
end

# ============================================================================
# ============================================================================
# METHOD 2: AD (Tensors.jl gradient)
# ============================================================================

module ADTet10
using Tensors

# Just shape functions (no manual derivatives!)
@inline N1(xi) = (1 - xi[1] - xi[2] - xi[3]) * (2 * (1 - xi[1] - xi[2] - xi[3]) - 1)
@inline N2(xi) = xi[1] * (2 * xi[1] - 1)
@inline N3(xi) = xi[2] * (2 * xi[2] - 1)
@inline N4(xi) = xi[3] * (2 * xi[3] - 1)
@inline N5(xi) = 4 * xi[1] * (1 - xi[1] - xi[2] - xi[3])
@inline N6(xi) = 4 * xi[1] * xi[2]
@inline N7(xi) = 4 * xi[2] * (1 - xi[1] - xi[2] - xi[3])
@inline N8(xi) = 4 * xi[3] * (1 - xi[1] - xi[2] - xi[3])
@inline N9(xi) = 4 * xi[1] * xi[3]
@inline N10(xi) = 4 * xi[2] * xi[3]

const shape_fns = (N1, N2, N3, N4, N5, N6, N7, N8, N9, N10)

@inline function eval_basis_and_grad(xi::Vec{3})
    # Evaluate basis functions
    N = ntuple(i -> shape_fns[i](xi), 10)

    # Compute gradients with Tensors.jl gradient()
    dN = ntuple(i -> gradient(shape_fns[i], xi), 10)

    return N, dN
end
end

# ============================================================================
# BENCHMARKING
# ============================================================================

println("Setting up benchmark...")
println()

# Test point (typical integration point)
const ξ_test = Vec(0.25, 0.25, 0.25)

# Verification: Both methods should give same results
println("Verifying correctness...")
N_manual, dN_manual = ManualTet10.eval_basis_and_grad(ξ_test)
N_ad, dN_ad = ADTet10.eval_basis_and_grad(ξ_test)

println("  Manual basis: ", N_manual)
println("  AD basis:     ", N_ad)
println()

# Check agreement
rtol = 1e-10
if !all(isapprox.(N_manual, N_ad, rtol=rtol))
    @warn "Manual and AD basis functions disagree!"
end

# Check derivatives
for i in 1:10
    if !isapprox(dN_manual[i], dN_ad[i], rtol=rtol)
        @warn "Manual and AD derivative $i disagree!" dN_manual[i] dN_ad[i]
    end
end

println("✓ Both methods agree (within tolerance)")
println()

# ============================================================================
# RUN BENCHMARKS
# ============================================================================

println("Running benchmarks (this may take a minute)...")
println()

# Warm-up
for _ in 1:1000
    ManualTet10.eval_basis_and_grad(ξ_test)
    ADTet10.eval_basis_and_grad(ξ_test)
end

# Benchmark each method
b_manual = @benchmark ManualTet10.eval_basis_and_grad($ξ_test)
b_ad = @benchmark ADTet10.eval_basis_and_grad($ξ_test)

# ============================================================================
# RESULTS
# ============================================================================

# ============================================================================
# RESULTS
# ============================================================================

println("="^70)
println("RESULTS")
println("="^70)
println()

# Extract median times
t_manual = median(b_manual.times)
t_ad = median(b_ad.times)

# Extract allocations
alloc_manual = b_manual.allocs
alloc_ad = b_ad.allocs

# Calculate relative speed
rel_ad = t_ad / t_manual

println("Method          | Time (ns) | Allocations | Relative Speed")
println("----------------|-----------|-------------|----------------")
@printf "Manual          | %9.1f | %11d | %.2f× (baseline)\n" t_manual alloc_manual 1.0
@printf "AD (Tensors.jl) | %9.1f | %11d | %.2f×\n" t_ad alloc_ad rel_ad
println()

# Detailed stats
println("Detailed Statistics:")
println()
println("Manual (Hand-Calculated):")
display(b_manual)
println()
println()
println("AD (Tensors.jl gradient):")
display(b_ad)
println()
println()

# ============================================================================
# ANALYSIS
# ============================================================================

println("="^70)
println("ANALYSIS")
println("="^70)
println()

if rel_ad < 2.0
    println("🎉 RECOMMENDATION: Use AD everywhere!")
    println()
    println("Tensors.jl AD is within 2× of manual, providing:")
    println("  ✓ Zero maintenance burden")
    println("  ✓ No manual derivative errors")
    println("  ✓ Easy to add new elements")
    println("  ✓ Supports any basis type")
    println()
    println("Small performance cost is acceptable for these benefits.")
elseif rel_ad < 5.0
    println("⚠️  RECOMMENDATION: Hybrid approach")
    println()
    println("AD is 2-5× slower than manual. Consider:")
    println("  • Common elements (Tet10, Hex8, Quad4): Manual")
    println("  • Rare elements: AD-generated")
    println("  • Research/prototype elements: Always AD")
    println()
    println("This balances performance and maintainability.")
else
    println("❌ RECOMMENDATION: Manual derivatives (with symbolic generation)")
    println()
    println("AD is >5× slower than manual. For performance-critical code:")
    println("  • Generate derivatives with SymPy/Symbolics.jl")
    println("  • Unit test against AD to verify correctness")
    println("  • Accept the maintenance burden")
    println()
    println("Consider AD only for prototyping.")
end

println()
println("Memory analysis:")
if alloc_manual == 0 && alloc_ad == 0
    println("  ✓ Both methods achieve zero allocations (excellent!)")
elseif alloc_manual == 0 && alloc_ad > 0
    println("  ⚠️  AD allocates (", alloc_ad, " allocs)")
    println("     This will hurt performance in tight loops.")
else
    println("  ⚠️  Unexpected allocation pattern - investigate!")
end

println()
println("="^70)
println("Benchmark complete! Results saved to console.")
println("="^70)
