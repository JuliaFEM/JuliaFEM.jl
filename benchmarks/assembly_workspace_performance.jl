# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Benchmark comparison between AssemblyMaterialWorkspaceMechanics and AssemblyMaterialWorkspace.

Tests:
1. Field access performance (workspace.σ[ip], workspace.𝔻[ip])
2. Vector extraction performance (get_tangent_vector)
3. Full assembly loop simulation
4. Allocations
"""

using JuliaFEM
using JuliaFEM: material_field_type, material_state_type, create_zero_state, AssemblyMaterialWorkspace, get_tangent_vector
using Tensors
using BenchmarkTools

# Setup
mat = LinearElastic(E=210e9, ν=0.3)
nips = 8
nelems = 1000  # Test with many elements to see accumulated overhead

# Create both workspace types
println("="^70)
println("ASSEMBLY WORKSPACE PERFORMANCE BENCHMARK")
println("="^70)

# Mechanics struct (specialized)
ws_mechanics = JuliaFEM.create_material_cache(mat, nips)
println("\n[1] Created AssemblyMaterialWorkspaceMechanics")
println("    Type: ", typeof(ws_mechanics))

# Force creation of general struct for comparison
# We need to create a material that doesn't match (σ, 𝔻) exactly
# Actually, let's manually create the general struct to compare
FieldType = JuliaFEM.material_field_type(mat)
StateType = JuliaFEM.material_state_type(mat)

# Create general struct manually
field_names = fieldnames(FieldType)
field_types = [fieldtype(FieldType, name) for name in field_names]
vecs = Vector[]
vec_names = Symbol[]
for (name, T) in zip(field_names, field_types)
    vec_name = Symbol("$(name)_vec")
    vec = [zero(T) for _ in 1:nips]
    push!(vecs, vec)
    push!(vec_names, vec_name)
end
field_vectors = NamedTuple{tuple(vec_names...)}(tuple(vecs...))
zero_state = JuliaFEM.create_zero_state(StateType)
states = [zero_state for _ in 1:nips]
ws_general = JuliaFEM.AssemblyMaterialWorkspace{FieldType, StateType}(field_vectors, states)

println("\n[2] Created AssemblyMaterialWorkspace (general)")
println("    Type: ", typeof(ws_general))

# ============================================================================
# Benchmark 1: Field Access (workspace.σ[ip], workspace.𝔻[ip])
# ============================================================================

println("\n" * "="^70)
println("BENCHMARK 1: Field Access (workspace.σ[ip], workspace.𝔻[ip])")
println("="^70)

function test_field_access_mechanics(ws, nips, nelems)
    for _ in 1:nelems
        for ip in 1:nips
            σ = ws.σ[ip]
            𝔻 = ws.𝔻[ip]
        end
    end
end

function test_field_access_general(ws, nips, nelems)
    for _ in 1:nelems
        for ip in 1:nips
            σ = ws.σ[ip]
            𝔻 = ws.𝔻[ip]
        end
    end
end

# Warm up
for _ in 1:100
    test_field_access_mechanics(ws_mechanics, nips, 10)
    test_field_access_general(ws_general, nips, 10)
end

# Benchmark mechanics
println("\n[Mechanics Struct]")
bench_mechanics = @benchmark test_field_access_mechanics($ws_mechanics, $nips, $nelems)
println("  Time:     ", round(mean(bench_mechanics.times) / 1e6, digits=3), " ms")
println("  Allocs:   ", bench_mechanics.allocs, " bytes")
println("  Memory:   ", bench_mechanics.memory, " bytes")
println("  Per element: ", round(mean(bench_mechanics.times) / nelems / 1e6, digits=6), " ms")

# Benchmark general
println("\n[General Struct]")
bench_general = @benchmark test_field_access_general($ws_general, $nips, $nelems)
println("  Time:     ", round(mean(bench_general.times) / 1e6, digits=3), " ms")
println("  Allocs:   ", bench_general.allocs, " bytes")
println("  Memory:   ", bench_general.memory, " bytes")
println("  Per element: ", round(mean(bench_general.times) / nelems / 1e6, digits=6), " ms")

# Comparison
time_ratio = mean(bench_general.times) / mean(bench_mechanics.times)
alloc_diff = bench_general.allocs - bench_mechanics.allocs
println("\n[Comparison]")
println("  Time ratio (general/mechanics): ", round(time_ratio, digits=3), "x")
println("  Allocation difference: ", alloc_diff, " bytes")
if alloc_diff == 0 && time_ratio < 1.1
    println("  ✅ Performance difference is negligible!")
else
    println("  ⚠️  Performance difference detected")
end
alloc_diff = bench_general.allocs - bench_mechanics.allocs  # Store for summary
time_ratio = time_ratio  # Store for summary

# ============================================================================
# Benchmark 2: Vector Extraction (get_tangent_vector)
# ============================================================================

println("\n" * "="^70)
println("BENCHMARK 2: Vector Extraction (get_tangent_vector)")
println("="^70)

function test_vector_extraction_mechanics(ws, nips, nelems)
    for _ in 1:nelems
        𝔻_vec = JuliaFEM.get_tangent_vector(ws)
        for ip in 1:nips
            𝔻 = 𝔻_vec[ip]
        end
    end
end

function test_vector_extraction_general(ws, nips, nelems)
    for _ in 1:nelems
        𝔻_vec = JuliaFEM.get_tangent_vector(ws)
        for ip in 1:nips
            𝔻 = 𝔻_vec[ip]
        end
    end
end

# Warm up
for _ in 1:100
    test_vector_extraction_mechanics(ws_mechanics, nips, 10)
    test_vector_extraction_general(ws_general, nips, 10)
end

# Benchmark mechanics
println("\n[Mechanics Struct]")
bench2_mechanics = @benchmark test_vector_extraction_mechanics($ws_mechanics, $nips, $nelems)
println("  Time:     ", round(mean(bench2_mechanics.times) / 1e6, digits=3), " ms")
println("  Allocs:   ", bench2_mechanics.allocs, " bytes")
println("  Memory:   ", bench2_mechanics.memory, " bytes")
println("  Per element: ", round(mean(bench2_mechanics.times) / nelems / 1e6, digits=6), " ms")

# Benchmark general
println("\n[General Struct]")
bench2_general = @benchmark test_vector_extraction_general($ws_general, $nips, $nelems)
println("  Time:     ", round(mean(bench2_general.times) / 1e6, digits=3), " ms")
println("  Allocs:   ", bench2_general.allocs, " bytes")
println("  Memory:   ", bench2_general.memory, " bytes")
println("  Per element: ", round(mean(bench2_general.times) / nelems / 1e6, digits=6), " ms")

# Comparison
time_ratio2 = mean(bench2_general.times) / mean(bench2_mechanics.times)
alloc_diff2 = bench2_general.allocs - bench2_mechanics.allocs
println("\n[Comparison]")
println("  Time ratio (general/mechanics): ", round(time_ratio2, digits=3), "x")
println("  Allocation difference: ", alloc_diff2, " bytes")
if alloc_diff2 == 0 && time_ratio2 < 1.1
    println("  ✅ Performance difference is negligible!")
else
    println("  ⚠️  Performance difference detected")
end
alloc_diff2 = bench2_general.allocs - bench2_mechanics.allocs  # Store for summary
time_ratio2 = time_ratio2  # Store for summary

# ============================================================================
# Benchmark 3: Full Assembly Loop Simulation
# ============================================================================

println("\n" * "="^70)
println("BENCHMARK 3: Full Assembly Loop Simulation")
println("="^70)

function test_assembly_loop_mechanics(ws, nips, nelems)
    # Simulate typical assembly loop
    for _ in 1:nelems
        𝔻_vec = JuliaFEM.get_tangent_vector(ws)
        for k in 1:3, l in k:3
            for ip in 1:nips
                C = 𝔻_vec[ip]
                # Simulate some computation
                _ = C
            end
        end
    end
end

function test_assembly_loop_general(ws, nips, nelems)
    # Simulate typical assembly loop
    for _ in 1:nelems
        𝔻_vec = JuliaFEM.get_tangent_vector(ws)
        for k in 1:3, l in k:3
            for ip in 1:nips
                C = 𝔻_vec[ip]
                # Simulate some computation
                _ = C
            end
        end
    end
end

# Warm up
for _ in 1:100
    test_assembly_loop_mechanics(ws_mechanics, nips, 10)
    test_assembly_loop_general(ws_general, nips, 10)
end

# Benchmark mechanics
println("\n[Mechanics Struct]")
bench3_mechanics = @benchmark test_assembly_loop_mechanics($ws_mechanics, $nips, $nelems)
println("  Time:     ", round(mean(bench3_mechanics.times) / 1e6, digits=3), " ms")
println("  Allocs:   ", bench3_mechanics.allocs, " bytes")
println("  Memory:   ", bench3_mechanics.memory, " bytes")
println("  Per element: ", round(mean(bench3_mechanics.times) / nelems / 1e6, digits=6), " ms")

# Benchmark general
println("\n[General Struct]")
bench3_general = @benchmark test_assembly_loop_general($ws_general, $nips, $nelems)
println("  Time:     ", round(mean(bench3_general.times) / 1e6, digits=3), " ms")
println("  Allocs:   ", bench3_general.allocs, " bytes")
println("  Memory:   ", bench3_general.memory, " bytes")
println("  Per element: ", round(mean(bench3_general.times) / nelems / 1e6, digits=6), " ms")

# Comparison
time_ratio3 = mean(bench3_general.times) / mean(bench3_mechanics.times)
alloc_diff3 = bench3_general.allocs - bench3_mechanics.allocs
println("\n[Comparison]")
println("  Time ratio (general/mechanics): ", round(time_ratio3, digits=3), "x")
println("  Allocation difference: ", alloc_diff3, " bytes")
if alloc_diff3 == 0 && time_ratio3 < 1.1
    println("  ✅ Performance difference is negligible!")
else
    println("  ⚠️  Performance difference detected")
end
alloc_diff3 = bench3_general.allocs - bench3_mechanics.allocs  # Store for summary
time_ratio3 = time_ratio3  # Store for summary

# ============================================================================
# Summary
# ============================================================================

println("\n" * "="^70)
println("SUMMARY")
println("="^70)
println("""
Benchmark Results:
  1. Field Access:        $(round(time_ratio, digits=3))x speed, $(alloc_diff) bytes diff
  2. Vector Extraction:  $(round(time_ratio2, digits=3))x speed, $(alloc_diff2) bytes diff
  3. Assembly Loop:      $(round(time_ratio3, digits=3))x speed, $(alloc_diff3) bytes diff

Recommendation:
""")

# Calculate per-element overhead
total_allocs_per_elem = max(alloc_diff, alloc_diff2, alloc_diff3) / nelems
time_overhead_per_elem = (mean(bench3_general.times) - mean(bench3_mechanics.times)) / nelems / 1e6  # ms

println("""
Per-element overhead (general vs mechanics):
  Time:     $(round(time_overhead_per_elem, digits=9)) ms per element
  Allocs:   $(round(total_allocs_per_elem, digits=3)) bytes per element
""")

# Decision threshold: if overhead is < 1ns per element and < 1 byte per element, remove specialization
if total_allocs_per_elem < 1.0 && time_overhead_per_elem < 1e-6
    println("  ✅ REMOVE AssemblyMaterialWorkspaceMechanics specialization")
    println("  ✅ Overhead is negligible: $(round(time_overhead_per_elem * 1e6, digits=3)) ns/elem, $(round(total_allocs_per_elem, digits=3)) bytes/elem")
    println("  ✅ Use only AssemblyMaterialWorkspace (simpler code, same performance)")
else
    println("  ⚠️  KEEP AssemblyMaterialWorkspaceMechanics (performance benefit detected)")
    println("  ⚠️  Time overhead: $(round(time_overhead_per_elem * 1e6, digits=3)) ns per element")
    println("  ⚠️  Allocation overhead: $(round(total_allocs_per_elem, digits=3)) bytes per element")
end

println("\n" * "="^70)
