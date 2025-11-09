---
title: "Type-Stable Field Storage for Performance"
description: "Why type stability matters for CPU, GPU, and MPI performance in FEM codes"
date: "November 9, 2025"
author: "Jukka Aho"
categories: ["architecture", "performance", "design"]
keywords: ["type-stability", "zero-allocation", "GPU", "MPI", "performance"]
audience: "researchers"
level: "advanced"
type: "design-rationale"
series: "The JuliaFEM Book"
chapter: 5
status: "proposal"
---

## Executive Summary

The v0.5.1 implementation used `Dict{String,Any}` for field storage, causing **9-92× performance degradation** compared to type-stable alternatives. More critically, type-unstable code **cannot run on GPUs** and incurs significant overhead in MPI communication.

**This document does NOT prescribe where field data should live** (element-local, global arrays, or elsewhere). Instead, it demonstrates why **type stability at access points** is essential for:

1. **CPU performance:** 9-92× speedup, zero allocations
2. **GPU execution:** Type-stable code can run in CUDA kernels
3. **MPI efficiency:** Contiguous, typed data transfers efficiently
4. **Threading:** Immutable access patterns enable parallelism

**Key Finding:** The choice of storage pattern matters less than ensuring type stability. Whether fields live in elements, element sets, or global structures, the access pattern must allow compile-time type inference.

---

## The Problem: Type Instability

### What is Type Instability?

When Julia cannot determine the type of a value at compile time, it must use **runtime dispatch**—looking up methods dynamically. This is slow and prevents optimization.

```julia
# Type-unstable: return type depends on runtime string value
function get_field(dict::Dict{String,Any}, name::String)
    return dict[name]  # Compiler sees: return type = Any
end

# Type-stable: return type known at compile time
function get_modulus(fields::NamedTuple)::Float64
    return fields.youngs_modulus  # Compiler sees: return type = Float64
end
```

### Why It Matters

1. **CPU:** Runtime dispatch is 10-100× slower than direct access
2. **GPU:** CUDA kernels **cannot** contain type-unstable code (compilation fails)
3. **MPI:** Sending `Any` types requires serialization; typed arrays use fast memcpy
4. **Optimization:** Compiler cannot inline, vectorize, or fuse operations through `Any`

### Measured Impact (v0.5.1)

Running `benchmarks/field_storage_comparison.jl` shows:

| Operation | Dict{String,Any} | Type-Stable | Speedup |
|-----------|------------------|-------------|---------|
| Field access | 19.2ns | 2.1ns | 9× |
| Nodal access | 262ns, 3 allocs | 6.5ns, 0 allocs | 40× |
| Interpolation | 2.6μs, 50 allocs | 53ns, 0 allocs | 49× |
| Assembly (1000 elem) | 109μs, 4000 allocs | 1.2μs, 0 allocs | 92× |

**Critical:** The cached interpolation and assembly loop achieve **zero allocations**. This is the performance required for GPU kernels.

---

## Design Requirements

Any field storage solution must satisfy:

### 1. Type Stability at Access Points

The compiler must infer types, regardless of where data is stored:

```julia
# ✅ Good: Type inferred from structure
E = fields.youngs_modulus  # Compiler knows: Float64

# ❌ Bad: Type depends on runtime value  
E = fields["youngs_modulus"]  # Compiler knows: Any
```

**Note:** This does NOT mean fields must be in a specific container. They could be in:

- Element-local storage: `element.youngs_modulus`
- Global arrays: `material_properties[element_set_id].E`
- Passed as arguments: `assemble!(element, E, ν, cache)`

What matters is that the access pattern is type-stable.

### 2. Zero Allocations in Hot Paths

GPU kernels and MPI communication require pre-allocated buffers:

```julia
# Assembly loop (hot path):
for element in elements
    # Must allocate nothing here ↓
    K_local = assemble_element(element, E, ν, cache)
    add_to_global!(K, element, K_local)
end
```

Allocations in inner loops destroy performance and prevent GPU execution.

### 3. Contiguous Memory Layout

GPU and MPI work best with contiguous arrays:

```julia
# ✅ Good: Contiguous, CUDA can transfer directly
displacement = Matrix{Float64}(3, n_nodes)  # AoS or SoA, both OK

# ⚠️ Problematic: Scattered, needs gathering for transfer
elements_with_fields = [(elem1, Dict("u" => ...)), (elem2, Dict("u" => ...))]
```

### 4. Immutable Where Possible

Immutable data enables safe parallelism (threading, GPU, MPI):

```julia
# ✅ Good: Can read from multiple threads safely
struct ConstantField{T}
    value::T  # Immutable
end

# ⚠️ Caution: Mutable data needs synchronization
mutable struct MutableField{T}
    value::T  # Requires locks for thread safety
end
```

---

## Demonstrated Solutions

The following are **examples**, not mandates. The key principle is type stability, not a specific implementation.

### Example 1: NamedTuple Container

**Approach:** Group related fields in a type-stable tuple.

```julia
fields = (
    E = 210e3,
    ν = 0.3,
    u = zeros(3, n_nodes),
)

# Access is type-stable:
modulus = fields.E  # Float64, inferred at compile time
```

**Pros:** Simple, type-stable, immutable  
**Cons:** Cannot add fields dynamically (but do we need to?)

See `benchmarks/field_storage_comparison.jl` for performance validation.

### Example 2: Struct with Typed Fields

**Approach:** Define problem-specific structs.

```julia
struct ElasticityFields
    E::Float64
    ν::Float64
    u::Matrix{Float64}
end

fields = ElasticityFields(210e3, 0.3, zeros(3, n_nodes))
modulus = fields.E  # Type-stable access
```

**Pros:** Explicit, self-documenting, type-stable  
**Cons:** Less flexible (but flexibility has a cost!)

### Example 3: Passed as Arguments

**Approach:** Don't store fields in containers at all—pass them explicitly.

```julia
function assemble_element(element, E::Float64, ν::Float64, u::Matrix{Float64}, cache)
    # All arguments have concrete types
    # Compiler can inline, optimize, specialize
end

# Caller provides fields:
for element in elements
    assemble_element(element, 210e3, 0.3, displacement, cache)
end
```

**Pros:** Maximum type stability, explicit dependencies  
**Cons:** Many arguments (but kwargs help: `assemble!(; E, ν, u)`)

---

## GPU and MPI: Why Type Stability is Essential

### GPU Execution

CUDA kernels require **all code to be type-stable**. The GPU compiler must generate specialized machine code—it cannot handle `Any` types.

**Demonstration:** See `benchmarks/gpu_mpi_mock.jl` for a mock CUDA kernel that:

1. Copies typed field data to GPU (fast memcpy)
2. Runs assembly kernel with zero allocations
3. Copies result back to CPU

The key insight: GPU code looks identical to optimized CPU code. Type stability enables both.

### MPI Communication

MPI transfers are fastest with contiguous, typed arrays:

```julia
# Fast: Typed array, direct buffer transfer
displacement = Matrix{Float64}(3, n_nodes)
MPI.Send(displacement, dest, tag, comm)

# Slow: Mixed types, requires serialization
fields = Dict{String,Any}("u" => displacement, "E" => 210e3)
MPI.send(fields, dest, tag, comm)  # Lowercase send = slow serialization
```

**Demonstration:** See `benchmarks/gpu_mpi_mock.jl` for MPI data transfer example.

---

## Recommendations (Not Requirements)

Based on benchmarks and GPU/MPI considerations:

1. **Use type-stable access patterns** wherever possible
2. **Prefer immutable data structures** to enable parallelism
3. **Pre-allocate caches** for hot-path operations
4. **Use contiguous arrays** for field data (enables fast GPU/MPI transfer)
5. **Profile with `@btime`** to verify zero allocations

**What we do NOT mandate:**

- Where field data must live (element, global, argument)
- Which container type to use (tuple, struct, separate arrays)
- When to use dynamic vs static structures

These are implementation details. Type stability is the non-negotiable requirement.

---

## Validation

Three benchmark scripts validate the claims:

1. **`benchmarks/field_storage_comparison.jl`**
   - CPU performance: 9-92× speedup
   - Zero allocations in hot paths
   - Run: `julia --project=. benchmarks/field_storage_comparison.jl`

2. **`benchmarks/gpu_mpi_mock.jl`**
   - GPU data transfer (mock, no CUDA dependency)
   - MPI communication patterns
   - Shows type-stable code can flow to GPU/MPI
   - Run: `julia --project=. benchmarks/gpu_mpi_mock.jl`

3. **`benchmarks/VALIDATION_RESULTS.md`**
   - Summary of measured results
   - Performance comparison table

---

## Conclusion

Type stability is not an implementation detail—it's a **fundamental requirement** for:

- High CPU performance (9-92× speedup measured)
- GPU execution (type-unstable code cannot compile for GPU)
- Efficient MPI communication (typed arrays transfer ~100× faster)
- Safe threading (immutable access patterns)

The v1.0 design must ensure type stability at field access points. The specific storage pattern (element-local, global arrays, or other) is a secondary concern that can be chosen based on other factors (memory layout, cache efficiency, user API).

**Next Steps:**

1. Review this rationale
2. Run benchmarks to verify claims
3. Choose storage pattern(s) appropriate for different use cases
4. Implement with type stability as the primary constraint
5. Validate GPU kernel execution with real CUDA code

---

**Last Updated:** November 9, 2025  
**Status:** Design rationale with validated performance measurements  
**Decision:** Type stability is mandatory; storage pattern is flexible
