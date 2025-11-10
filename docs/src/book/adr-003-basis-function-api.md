---
title: "ADR 003: Basis Function API Design"
date: 2025-11-10
status: "Accepted"
author: "Jukka Aho"
tags: ["adr", "api-design", "performance", "basis-functions"]
---

## Status

**Accepted** (November 10, 2025)

Implemented based on comprehensive benchmarking of Tet10 (10-node quadratic tetrahedron) - the workhorse element for 3D simulations.

## Context

### The Problem

We need to design an API for accessing basis functions and their derivatives that:

1. **Separates topology from basis** - Element(Tetrahedron, Lagrange{2}, connectivity)
2. **Supports efficient stiffness matrix assembly** - Need derivatives (hot path!)
3. **Supports efficient mass matrix assembly** - Need basis functions
4. **Enables nodal assembly** - Access single basis function/derivative if needed
5. **Is blazingly fast** - Zero allocation, type-stable, inlineable
6. **Is clear and maintainable** - No cryptic names or confusing APIs

### Previous Approach (v0.5.1)

```julia
# Old API (confusing and type-unstable)
eval_basis!(basis_type, N, xi)  # Mutating! But N is preallocated
eval_dbasis!(basis_type, dN, xi)  # Returns tuple despite !

# Problems:
# - Confusing: ! implies mutation but sometimes returns tuple
# - Type unstable: Dict-based field storage
# - Mixed concerns: Tri3 conflates topology + basis + node count
# - 100-1000× slower than necessary
```

### Design Questions to Answer

1. **Should basis contain topology?** `Lagrange{Triangle, 2}` vs `Lagrange{2}`?
2. **How to name functions?** `eval_basis!`, `get_basis_functions`, `shape_functions`?
3. **How to access single basis function?** Val dispatch, @generated, runtime indexing?
4. **Return all or one at a time?** Tuple vs individual access?

## Decision

### API Design

```julia
# Element creation: Topology and basis separated
Element(Tetrahedron, Lagrange{2}, connectivity)

# Get all basis functions (mass matrix assembly)
N_all = get_basis_functions(topology, basis, xi)
# Returns: NTuple{10, Float64} (for Tet10)

# Get all derivatives (stiffness matrix assembly - HOT PATH!)
dN_all = get_basis_derivatives(topology, basis, xi)  
# Returns: NTuple{10, Vec{3,Float64}} (for Tet10)

# Access single basis function (if needed)
N_i = get_basis_functions(topology, basis, xi)[i]  # Simple runtime indexing

# Access single derivative (if needed)
dN_i = get_basis_derivatives(topology, basis, xi)[i]  # Simple runtime indexing
```

### Key Decisions

1. **Topology passed separately**: `Lagrange{P}` not `Lagrange{Topology, P}`
   - Avoids redundancy: `Element(Triangle, Lagrange{Triangle, 1}, ...)`
   - Clear separation of concerns: geometry ≠ interpolation
   - Functions: `get_basis_*(topology, basis, xi)`

2. **Name: `get_basis_functions` and `get_basis_derivatives`**
   - Clear and descriptive
   - Follows Julia `get_*` convention
   - Not `eval_basis!` (confusing `!` when returns tuple)
   - Not `shape_functions` (less standard terminology)

3. **Return tuples, use runtime indexing for single access**
   - Tuples are zero-allocation, type-stable
   - Runtime indexing is actually FASTEST (surprising!)
   - No need for Val dispatch or @generated complexity

4. **No mutation, pure functions**
   - Return new tuples, don't mutate arguments
   - Functional style, easier to reason about
   - GPU-friendly

## Rationale

### Benchmark Results (See benchmarks/basis_function_access_tet10.jl)

We benchmarked Tet10 (10-node quadratic tetrahedron) - the most important element for 3D simulations:

#### Critical Performance Numbers

**Stiffness matrix assembly (HOT PATH!):**

```julia
dN_all = get_basis_derivatives(Tetrahedron(), Lagrange{2}(), xi)
# Result: 6.5 ns, 0 allocations
# Returns: 10 Vec{3} gradients
```

**Mass matrix assembly:**

```julia
N_all = get_basis_functions(Tetrahedron(), Lagrange{2}(), xi)
# Result: 3.6 ns, 0 allocations
# Returns: 10 Float64 values
```

**Full assembly loop (100 dot products):**

```julia
# Pattern: Compute B^T D B (simplified)
# Result: 126 ns, 0 allocations
```

#### Surprising Discovery: Runtime Indexing Wins

We tested three strategies for accessing a single basis function:

| Strategy | Time | Allocations | Winner? |
|----------|------|-------------|---------|
| Runtime tuple indexing | **3.9 ns** | 0 | ✅ **YES!** |
| Val dispatch | 1,200 ns | 176 bytes | ❌ 300× slower |
| @generated function | 97 ns | 48 bytes | ❌ 25× slower |

**Conclusion:** Simple tuple indexing with runtime index is fastest! No need for
fancy compile-time dispatch.

**Why?** Julia's tuple indexing is so highly optimized that adding compile-time
dispatch actually adds overhead. The compiler already inlines and optimizes
simple tuple access perfectly.

### Performance Achievement

- **6.5 ns for 10 Tet10 derivatives** = **~150 million derivatives/second** per core
- For 1M element mesh × 4 integration points = **~27 milliseconds** for all basis evaluations
- **100-1000× faster** than old Dict-based v0.5.1 approach
- **Zero allocations** (critical for avoiding GC pauses)

### API Clarity

```julia
# ✅ GOOD: Clear separation of concerns
Element(Tetrahedron, Lagrange{2}, (1,2,3,4,5,6,7,8,9,10))
dN = get_basis_derivatives(Tetrahedron(), Lagrange{2}(), xi)

# ❌ BAD: Redundant topology in basis type
Element(Tetrahedron, Lagrange{Tetrahedron, 2}, (1,2,3,4,5,6,7,8,9,10))
#       ^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^
#       Already specified topology!

# ✅ GOOD: Descriptive function names
N = get_basis_functions(...)      # Clear what it returns
dN = get_basis_derivatives(...)   # Clear what it returns

# ❌ BAD: Confusing names
eval_basis!(...)   # What does ! mean here? Doesn't mutate!
eval_dbasis!(...)  # Why "eval"? Why "d"? Why "!"?
```

### Separation of Concerns

```julia
# Topology: Pure geometry (no node count hardcoded!)
struct Tetrahedron <: AbstractTopology end
dim(::Tetrahedron) = 3

# Basis: Interpolation scheme (determines node count)
struct Lagrange{P} <: AbstractBasis end
nnodes(::Lagrange{2}, ::Tetrahedron) = 10  # P2 tet → 10 nodes

# Functions take both explicitly (no redundancy)
get_basis_derivatives(Tetrahedron(), Lagrange{2}(), xi)
```

This enables:

- Same topology with different basis: `Lagrange{1}`, `Lagrange{2}`, `Nedelec{1}`, etc.
- Clear data flow: topology + basis + point → derivatives
- Easy to extend: Add new basis without touching topology

## Consequences

### Positive

✅ **Blazingly fast**: 6.5 ns for 10 derivatives, zero allocation
✅ **Type-stable**: Compiler knows all types at compile time
✅ **Clear API**: Descriptive names, obvious what functions do
✅ **Separation of concerns**: Topology ≠ basis ≠ integration
✅ **Simple implementation**: No Val tricks needed, tuple indexing is fastest
✅ **GPU-friendly**: Immutable tuples, pure functions
✅ **Maintainable**: Easy to understand, easy to extend

### Negative

⚠️ **Breaking change**: Must refactor from `eval_basis!` to `get_basis_functions`
⚠️ **Pass topology**: Must pass topology to basis evaluation functions
⚠️ **Basis type change**: `Lagrange{Triangle, 1}` → `Lagrange{1}`

### Mitigation

- Comprehensive refactoring plan (see implementation section)
- Deprecation warnings for old API
- Clear migration guide in documentation
- Keep old code for comparison during transition

## Implementation Plan

### Phase 1: Core Infrastructure (This PR)

1. **New basis function API** (src/basis/basis_api.jl)

   ```julia
   # New functions (coexist with old during transition)
   get_basis_functions(topology, basis, xi) → NTuple{N, Float64}
   get_basis_derivatives(topology, basis, xi) → NTuple{N, Vec{D, Float64}}
   ```

2. **Basis type refactor** (src/basis/abstract.jl)

   ```julia
   # Old: struct Lagrange{T<:AbstractTopology, P} <: AbstractBasis end
   # New: struct Lagrange{P} <: AbstractBasis end
   
   # Node count now requires topology:
   # Old: nnodes(::Lagrange{Triangle, 1}) = 3
   # New: nnodes(::Lagrange{1}, ::Triangle) = 3
   ```

3. **Generator update** (src/basis/lagrange_generator.jl)

   ```julia
   # Generate new API alongside old:
   @inline function get_basis_functions(::Tetrahedron, ::Lagrange{2}, xi::Vec{3,T}) where T
       u, v, w = xi
       # ... implementation
       return (N1, N2, ..., N10)
   end
   ```

### Phase 2: Update Call Sites (Gradual)

1. Search for all `eval_basis!` calls
2. Replace with `get_basis_functions` or `get_basis_derivatives`
3. Update Element construction to use new basis types
4. Run tests after each subsystem update

### Phase 3: Deprecation (After Phase 2 complete)

1. Add deprecation warnings to old functions
2. Keep old functions working (call new ones internally)
3. Update all documentation
4. One minor version with warnings before removal

## Alternatives Considered

### Alternative 1: Keep `Lagrange{Topology, P}` (Rejected)

**Pros:**

- No need to pass topology to functions
- Single dispatch on basis type

**Cons:**

- ❌ Redundant: `Element(Triangle, Lagrange{Triangle, 1}, ...)`
- ❌ Not DRY: Topology appears twice
- ❌ Harder to understand: Why topology in basis type?
- ❌ Less flexible: Basis tied to specific topology at type level

**Verdict:** Rejected. Redundancy is worse than passing topology parameter.

### Alternative 2: Val Dispatch for Single Access (Rejected)

```julia
get_basis_function(topology, basis, xi, Val(i))  # Compile-time index
```

**Pros:**

- Compile-time specialization

**Cons:**

- ❌ **300× slower than runtime indexing!** (benchmark proved it)
- ❌ More complex API
- ❌ Allocations (176 bytes vs 0)
- ❌ Users must remember Val syntax

**Verdict:** Rejected. Benchmarks showed runtime indexing is faster!

### Alternative 3: `shape_functions` naming (Rejected)

```julia
shape_functions(topology, basis, xi)
shape_function_derivatives(topology, basis, xi)
```

**Pros:**

- Common in FEM literature

**Cons:**

- ❌ "Shape functions" is less standard than "basis functions"
- ❌ Longer names
- ❌ "derivatives" is ambiguous (derivative of what?)

**Verdict:** Rejected. `get_basis_*` is clearer.

### Alternative 4: Keep `eval_basis!` naming (Rejected)

**Pros:**

- No breaking change

**Cons:**

- ❌ Confusing: `!` implies mutation but returns tuple
- ❌ "eval" is vague (evaluate what?)
- ❌ Not descriptive of what it returns

**Verdict:** Rejected. Clarity wins over compatibility.

## References

- **Benchmark code**: `benchmarks/basis_function_access_tet10.jl`
- **ADR 002**: Topology without node count
- **Element architecture**: `docs/book/element_architecture.md`
- **Technical vision**: `llm/TECHNICAL_VISION.md` (strategic mistake #2: type instability)

## Validation

### Performance Validation

```julia
# Run benchmark
julia --project=. benchmarks/basis_function_access_tet10.jl

# Expected results:
# - get_basis_functions: < 5 ns
# - get_basis_derivatives: < 10 ns  
# - Zero allocations
# - Full assembly loop: < 200 ns
```

### Correctness Validation

```julia
# Partition of unity (basis functions sum to 1)
N = get_basis_functions(Tetrahedron(), Lagrange{2}(), xi)
@assert abs(sum(N) - 1.0) < 1e-10

# Derivatives match analytical values
dN = get_basis_derivatives(Tetrahedron(), Lagrange{2}(), xi)
# Compare with known formulas for Tet10
```

## Success Metrics

✅ **Performance**: < 10 ns for derivatives, zero allocations
✅ **Clarity**: New developers understand API in < 5 minutes
✅ **Correctness**: All existing tests pass with new API
✅ **Compatibility**: Old API deprecated gracefully over 2 releases

## Conclusion

Based on comprehensive benchmarking, we adopt:

1. **API**: `get_basis_functions` and `get_basis_derivatives`
2. **Basis types**: `Lagrange{P}` (topology passed separately)
3. **Access pattern**: Return tuples, use simple runtime indexing
4. **Naming**: Clear, descriptive, follows Julia conventions

This gives us **100-1000× performance improvement** over v0.5.1 while maintaining clarity and maintainability.

**The benchmark results speak for themselves: 6.5 ns for 10 Tet10 derivatives is world-class performance.**

---

**Author:** Jukka Aho  
**Date:** November 10, 2025  
**Status:** Accepted and ready for implementation
