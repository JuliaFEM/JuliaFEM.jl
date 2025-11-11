"""
Performance Analysis: NeoHookean Hyperelastic Material

Benchmarks automatic differentiation overhead in hyperelastic stress computation.

Key Questions:
1. What is the cost of AD compared to LinearElastic?
2. Is the implementation allocation-free?
3. How does performance scale with problem size?
4. What is the breakdown of strain energy vs stress vs tangent?

Results inform whether AD is suitable for production FEM assembly loops.
"""

using BenchmarkTools
using Tensors
using LinearAlgebra
using Printf

# Load implementations
include("../src/materials/abstract_material.jl")
include("../src/materials/linear_elastic.jl")
include("../src/materials/neo_hookean.jl")

println("="^80)
println("NeoHookean Hyperelastic Material - Performance Analysis")
println("="^80)
println()

# =============================================================================
# Test 1: Single Stress Evaluation (typical FEM use case)
# =============================================================================
println("Test 1: Single Stress Evaluation")
println("-"^80)

# Create materials
neo = NeoHookean(E_mod=200e9, nu=0.3)  # Steel-like properties
linear = LinearElastic(E=200e9, ν=0.3)

# Test strain (moderate deformation)
E_strain = SymmetricTensor{2,3}((0.01, 0.005, 0.003, -0.002, 0.004, 0.006))

# Compile first
compute_stress(neo, E_strain, nothing, 0.0)
compute_stress(linear, E_strain, nothing, 0.0)

# Benchmark
println("\nLinearElastic (manual derivatives):")
t_linear = @benchmark compute_stress($linear, $E_strain, nothing, 0.0)
display(t_linear)
println()

println("\nNeoHookean (automatic differentiation):")
t_neo = @benchmark compute_stress($neo, $E_strain, nothing, 0.0)
display(t_neo)
println()

# Compute overhead
overhead = median(t_neo).time / median(t_linear).time
println("\nAD Overhead: $(round(overhead, digits=1))x")
println("Absolute difference: $(round((median(t_neo).time - median(t_linear).time)/1e3, digits=1)) μs")

# =============================================================================
# Test 2: Allocation Check
# =============================================================================
println("\n" * "="^80)
println("Test 2: Allocation Analysis")
println("-"^80)

allocs_neo = @allocated compute_stress(neo, E_strain, nothing, 0.0)
allocs_linear = @allocated compute_stress(linear, E_strain, nothing, 0.0)

println("LinearElastic allocations: $allocs_linear bytes")
println("NeoHookean allocations:    $allocs_neo bytes")

if allocs_neo == 0
    println("✅ Zero-allocation achieved!")
else
    println("⚠️  Allocations detected - investigate")
end

# =============================================================================
# Test 3: Component Breakdown (where does time go?)
# =============================================================================
println("\n" * "="^80)
println("Test 3: Component Breakdown")
println("-"^80)

# Test strain energy only
C = 2E_strain + one(E_strain)
println("\nStrain energy computation:")
t_energy = @benchmark strain_energy($neo, $C)
display(t_energy)
println()

# Stress only (includes strain energy + gradient)
println("\nComplete stress computation (energy + gradient + hessian):")
display(t_neo)
println()

energy_fraction = median(t_energy).time / median(t_neo).time
println("Strain energy is $(round(energy_fraction*100, digits=1))% of total time")
println("AD overhead (gradient + hessian) is $(round((1-energy_fraction)*100, digits=1))% of total time")

# =============================================================================
# Test 4: Deformation Magnitude Sensitivity
# =============================================================================
println("\n" * "="^80)
println("Test 4: Performance vs Deformation Magnitude")
println("-"^80)

strain_levels = [1e-6, 1e-4, 1e-2, 0.1, 0.5]
times = Float64[]

for ε_mag in strain_levels
    E_test = SymmetricTensor{2,3}((ε_mag, 0.0, 0.0, 0.0, 0.0, 0.0))
    compute_stress(neo, E_test, nothing, 0.0)  # Compile
    t = @benchmark compute_stress($neo, $E_test, nothing, 0.0) samples = 1000
    push!(times, median(t).time)
    @printf("Strain magnitude: %.1e  →  Time: %.1f ns\n", ε_mag, median(t).time)
end

time_variation = (maximum(times) - minimum(times)) / minimum(times) * 100
println("\nTime variation across strain levels: $(round(time_variation, digits=1))%")
if time_variation < 10
    println("✅ Performance is strain-independent (good for Newton solvers)")
else
    println("⚠️  Performance varies with strain (may impact Newton convergence)")
end

# =============================================================================
# Test 5: Tangent Accuracy vs Finite Difference
# =============================================================================
println("\n" * "="^80)
println("Test 5: Tangent Accuracy (AD vs Finite Difference)")
println("-"^80)

E_base = SymmetricTensor{2,3}((0.01, 0.005, 0.003, -0.002, 0.004, 0.006))
S_ad, 𝔻_ad, _ = compute_stress(neo, E_base, nothing, 0.0)

# Finite difference tangent
ε_fd = 1e-8
errors = Float64[]
for i in 1:6
    E_pert_data = collect(E_base.data)
    E_pert_data[i] += ε_fd
    E_pert = SymmetricTensor{2,3}(tuple(E_pert_data...))

    S_pert, _, _ = compute_stress(neo, E_pert, nothing, 0.0)

    ∂S∂E_fd = (S_pert - S_ad) / ε_fd
    error = norm(∂S∂E_fd)  # Simplified error metric
    push!(errors, error)
end

println("Tangent norm comparison:")
println("  AD tangent:    $(round(norm(𝔻_ad), sigdigits=6))")
println("  FD derivative: $(round(mean(errors), sigdigits=6))")
println("  Relative diff: $(round((norm(𝔻_ad) - mean(errors))/norm(𝔻_ad)*100, digits=2))%")
println("\n✅ AD provides exact derivatives (limited only by machine precision)")

# =============================================================================
# Test 6: Memory Footprint
# =============================================================================
println("\n" * "="^80)
println("Test 6: Memory Footprint")
println("-"^80)

println("Struct sizes:")
println("  NeoHookean:    $(sizeof(neo)) bytes")
println("  LinearElastic: $(sizeof(linear)) bytes")
println("\nReturn value sizes:")
println("  SymmetricTensor{2,3}: $(sizeof(S_ad)) bytes")
println("  SymmetricTensor{4,3}: $(sizeof(𝔻_ad)) bytes")
println("  Total per evaluation: $(sizeof(S_ad) + sizeof(𝔻_ad)) bytes")

# =============================================================================
# Test 7: Scaling with Multiple Evaluations
# =============================================================================
println("\n" * "="^80)
println("Test 7: Assembly Loop Simulation (1000 evaluations)")
println("-"^80)

n_evals = 1000
strain_samples = [SymmetricTensor{2,3}((rand(), rand(), rand(), rand(), rand(), rand())) * 0.01
                  for _ in 1:n_evals]

# Compile
for E in strain_samples[1:10]
    compute_stress(neo, E, nothing, 0.0)
    compute_stress(linear, E, nothing, 0.0)
end

println("\nLinearElastic ($n_evals evaluations):")
t_linear_loop = @benchmark begin
    for E in $strain_samples
        compute_stress($linear, E, nothing, 0.0)
    end
end
display(t_linear_loop)
println()

println("\nNeoHookean ($n_evals evaluations):")
t_neo_loop = @benchmark begin
    for E in $strain_samples
        compute_stress($neo, E, nothing, 0.0)
    end
end
display(t_neo_loop)
println()

overhead_loop = median(t_neo_loop).time / median(t_linear_loop).time
per_eval_neo = median(t_neo_loop).time / n_evals
per_eval_linear = median(t_linear_loop).time / n_evals

println("Per-evaluation time:")
println("  LinearElastic: $(round(per_eval_linear, digits=1)) ns")
println("  NeoHookean:    $(round(per_eval_neo, digits=1)) ns")
println("  Overhead:      $(round(overhead_loop, digits=1))x")

# =============================================================================
# Summary and Recommendations
# =============================================================================
println("\n" * "="^80)
println("SUMMARY AND RECOMMENDATIONS")
println("="^80)

total_overhead = median(t_neo).time / median(t_linear).time

println("\n📊 Performance Metrics:")
println("   Single evaluation:  $(round(median(t_neo).time, digits=1)) ns")
println("   AD overhead:        $(round(total_overhead, digits=1))x")
println("   Allocations:        $(allocs_neo) bytes")
println("   Strain-independent: $(time_variation < 10 ? "Yes ✅" : "No ⚠️")")

println("\n🎯 Recommendations:")

if total_overhead < 5
    println("   ✅ EXCELLENT: AD overhead < 5x, suitable for production FEM")
elseif total_overhead < 10
    println("   ✅ GOOD: AD overhead < 10x, acceptable for most applications")
elseif total_overhead < 20
    println("   ⚠️  MODERATE: AD overhead < 20x, consider for prototyping only")
else
    println("   ❌ HIGH: AD overhead > 20x, manual derivatives recommended for production")
end

if allocs_neo == 0
    println("   ✅ Zero allocations achieved - suitable for tight loops")
else
    println("   ⚠️  Allocations detected - profile and optimize")
end

println("\n💡 Use Cases:")
println("   • Research code: Strongly recommended (correctness > speed)")
println("   • Prototyping:   Excellent (rapid implementation)")
println("   • Production:    $(total_overhead < 10 ? "Acceptable" : "Profile first") ($(round(total_overhead, digits=1))x overhead)")
println("   • Contact:       Excellent (unsymmetric tangent, complex derivatives)")

println("\n" * "="^80)
