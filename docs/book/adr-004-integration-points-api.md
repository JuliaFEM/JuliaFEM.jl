---
title: "ADR-004: Zero-Allocation Integration Points API"
date: 2025-11-11
author: "Jukka Aho + AI Assistant"
status: "Accepted"
tags: ["adr", "integration", "performance", "api-design"]
---

## Context

Integration points (Gauss quadrature) are accessed millions of times during FEM assembly. The original implementation used runtime dispatch with mutable structs containing Dict fields, causing:

1. **Type instability** - Dict fields unknown at compile time
2. **Allocations** - New struct allocations every query
3. **~50× performance penalty** vs optimal approach

**Problem Statement:** How should integration points be accessed in assembly loops?

## Decision

**Adopt compile-time integration point API matching basis function design (eval_basis!).**

### New API

```julia
@inline function get_gauss_points!(::Type{T}, ::Type{S}) where {T<:AbstractTopology, S<:Gauss}
    -> NTuple{N, Tuple{Float64, Vec{D}}}
```

**Key Properties:**

- **Compile-time generation**: Like `eval_basis!`, returns literal tuples
- **Vec{D} coordinates**: Tensors.jl Vec for efficient FEM math
- **Zero allocation**: Fully inlined, no runtime overhead
- **Type-stable**: All types known at compile time

### Usage Pattern

```julia
# Assembly loop - zero allocations:
for (weight, ξ) in get_gauss_points!(Triangle, Gauss{2})
    N = eval_basis!(Lagrange{Triangle,1}, Float64, ξ)
    dN = eval_dbasis!(Lagrange{Triangle,1}, ξ)
    detJ = compute_jacobian(ξ)
    K += weight * detJ * (dN' * D * dN)
end
```

**Why Vec{D}?**

- Natural for FEM: `dN/dξ ⋅ v`, tensor products, etc.
- GPU-friendly (immutable, stack-allocated)
- Matches golden standard (nodal assembly demos)

## Alternatives Considered

### Option A: Plain Tuples

```julia
get_gauss_points!(Triangle, Gauss{1}) 
# → ((0.5, (1/3, 1/3)),)
```

**Rejected:** Tuple coordinates less convenient for FEM math.

### Option B: Store in Element

```julia
struct Element{N,NIP,...}
    ips::NTuple{NIP, IntegrationPoint{D}}
end
```

**Rejected:** Slight overhead, less flexible (fixed at construction).

### Option C: Global Constants

```julia
const TRI3_GAUSS1_IPS = ((0.5, Vec{2}((1/3, 1/3))),)
```

**Rejected:** Not composable (can't parameterize on topology/order).

### Option D: Runtime Dispatch (OLD)

```julia
get_integration_points(element::Seg2)
# → Vector{IP}  # Mutable struct with Dict
```

**Rejected:** 50× slower, allocates, type-unstable.

## Performance Results

Benchmark: 1000 elements, 3 integration points each

| Approach | Time | Allocations | Speedup |
|----------|------|-------------|---------|
| OLD (runtime + Dict) | 53 μs | 515 KiB | 1× |
| NEW (compile-time + Vec) | **1.1 μs** | **0 bytes** | **48×** |

**Realistic FEM assembly:**

- OLD: 53 μs + 515 KiB allocations
- NEW: 1.1 μs + 0 allocations

## Implementation

### File Structure

```text
src/integration/
├── integration.jl      # Abstract types (IntegrationPoint, AbstractIntegration)
├── gauss.jl            # High-level Gauss{N} type
└── gauss_points.jl     # NEW: Compile-time get_gauss_points!()
```

### Supported Topologies

**1D:**

- Segment: Gauss{1}, Gauss{2}, Gauss{3}

**2D:**

- Triangle: Gauss{1} (1 pt), Gauss{2} (3 pt), Gauss{3} (4 pt)
- Quadrilateral: Gauss{1} (1 pt), Gauss{2} (4 pt), Gauss{3} (9 pt)

**3D:**

- Tetrahedron: Gauss{1} (1 pt), Gauss{2} (4 pt), Gauss{3} (5 pt)
- Hexahedron: Gauss{1} (1 pt), Gauss{2} (8 pt), Gauss{3} (27 pt)
- Wedge: Gauss{1}, Gauss{2}
- Pyramid: Gauss{1}, Gauss{2}

## Consequences

### Positive

1. **50× faster** than old approach
2. **Zero allocations** in assembly loops
3. **Type-stable** - compiler knows everything
4. **Consistent with basis API** - same pattern as `get_basis_functions` (NOTE: `eval_basis!` is deprecated)
5. **GPU-ready** - Vec{D} immutable, can transfer to GPU
6. **Matches golden standard** - nodal assembly architecture

### Negative

1. **Breaking change** - old `get_integration_points(element)` deprecated
2. **Migration needed** - update assembly code to new API
3. **More verbose** - must specify topology and scheme explicitly

### Neutral

1. **Compile-time only** - dynamic integration orders need workaround
2. **Fixed quadrature rules** - pre-defined Gauss{1}, Gauss{2}, etc.

## Migration Strategy

### Phase 1: Add New API (✅ Complete)

- Implement `get_gauss_points!()` for all topologies
- Comprehensive tests
- Benchmark validation

### Phase 2: Update Assembly Code (In Progress)

- Fix `get_integration_points(element)` in elements.jl
- Update problem assembly functions
- Ensure tests pass

### Phase 3: Deprecate Old API

- Add deprecation warnings to old functions
- Document migration path
- Remove after one release cycle

## Related

- **ADR-002:** Basis function API (same pattern)
- **Golden Standard:** docs/book/multigpu_nodal_assembly.md
- **Nodal Assembly Demos:** demos/nodal_assembly_{cpu,gpu}.jl

## References

1. Benchmark: `benchmarks/integration_points_benchmark.jl`
2. Tests: `test/test_integration_points_api.jl`
3. Implementation: `src/integration/gauss_points.jl`

## Status History

- 2025-11-11: Accepted, implemented, tested
- Performance validated: 48× speedup, zero allocations
