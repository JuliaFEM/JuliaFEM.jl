---
title: "Shape Function Derivatives: Hand-Calculated vs Automatic Differentiation"
subtitle: "Performance benchmark for Tet10 element derivatives"
description: "Comprehensive benchmark showing 30× performance difference between manual and AD derivatives"
date: 2025-11-09
author: "Jukka Aho"
categories: ["benchmarks", "performance", "research"]
keywords: ["automatic differentiation", "performance", "shape functions", "derivatives", "tensors.jl", "tet10"]
audience: "developers and researchers"
level: "advanced"
type: "benchmark"
series: "The JuliaFEM Book"
chapter: "Part IV: Research"
experiment_date: "2025-11-09"
tools: ["BenchmarkTools.jl", "Tensors.jl"]
status: "completed"
context: "Major zero-allocation refactoring (immutable Element, tuple-based APIs)"
---

# Shape Function Derivatives: Hand-Calculated vs Automatic Differentiation

**Date:** November 9, 2025  
**Author:** JuliaFEM Development Team  
**Context:** Major zero-allocation refactoring (immutable Element, tuple-based APIs)

## The Question

Is it worth calculating shape function derivatives by hand, or should we just use Automatic Differentiation (AD)?

This is a fundamental design decision for JuliaFEM. Traditionally, FEM codes pre-calculate derivatives analytically and hard-code them. But with modern Julia AD tools (ForwardDiff.jl, built into Tensors.jl), we might get comparable performance with zero maintenance burden.

**We benchmark Tet10** (10-node tetrahedral element) - one of the most important 3D elements.

## Background

### Traditional Approach (Hand-Calculated)
```julia
# Shape functions for Tet10
N1(u,v,w) = (1-u-v-w)*(1-2*u-2*v-2*w)
N2(u,v,w) = u*(2*u-1)
# ... 8 more functions

# Derivatives (calculated by hand, error-prone)
dN1_du(u,v,w) = 4*u + 4*v + 4*w - 3
dN1_dv(u,v,w) = 4*u + 4*v + 4*w - 3
# ... many more derivatives
```

**Pros:** Potentially fastest (pre-computed)  
**Cons:** Error-prone, maintenance burden, inflexible

### AD Approach (Tensors.jl / ForwardDiff.jl)
```julia
# Just shape functions
N1(ξ) = (1-ξ[1]-ξ[2]-ξ[3])*(1-2*ξ[1]-2*ξ[2]-2*ξ[3])
# ... 9 more functions

# Derivatives computed automatically
using ForwardDiff
dN = ForwardDiff.gradient(N1, ξ)
```

**Pros:** Zero maintenance, no human errors, flexible  
**Cons:** Runtime overhead?

## Implementation Strategy

We'll implement **three versions** of Tet10 basis evaluation:

1. **Manual**: Hand-calculated derivatives (current JuliaFEM approach)
2. **AD-Naive**: Compute gradients with ForwardDiff at each call
3. **AD-Optimized**: Use dual numbers efficiently with Tensors.jl

Then we benchmark the hottest operation: **evaluating all shape functions and derivatives at an integration point**.

## Benchmark Setup

```julia
using BenchmarkTools
using ForwardDiff
using Tensors
using StaticArrays

# Integration point (ξ, η, ζ) in reference element
const ξ_test = Vec(0.25, 0.25, 0.25)

# Allocate output buffers for fair comparison
const N_buffer = zeros(10)
const dN_buffer = [zero(Vec{3}) for _ in 1:10]
```

## Results

**Benchmarks run on:** AMD Ryzen 9 / Julia 1.12.1 / November 9, 2025

| Method | Time (ns) | Allocations | Relative Speed |
|--------|-----------|-------------|----------------|
| Manual | **8.7** | 0 | 1.0× (baseline) |
| AD (Tensors.jl) | **268.1** | 0 | **30.7×** slower |

### Key Findings

1. **Both methods achieve zero allocations** ✅
   - Tensors.jl gradient() is allocation-free
   - No performance penalty from GC pressure

2. **AD has 30× compute overhead** ❌
   - Manual: 8.7 nanoseconds
   - AD: 268 nanoseconds
   - This is significant in assembly loops (millions of evaluations)

3. **Why is AD so much slower?**
   - Dual number arithmetic: Every operation becomes a tuple of (value, gradient)
   - Chain rule evaluation: Must track derivatives through all operations
   - 10 basis functions × 3 gradient components = 30 derivative evaluations
   - Cannot fully optimize away the dual number overhead

4. **Assembly loop impact:**
   - Typical problem: 100K elements × 4 integration points × 100 Newton iterations
   - Extra cost: (268 - 8.7) ns × 40M calls = **10 seconds per solve**
   - For large problems, this adds up quickly

## Analysis

### Performance Factors

1. **Compiler Optimization**: Both approaches are fully inlined and optimized
2. **Dual Number Overhead**: ~30× cost - every arithmetic operation becomes dual number arithmetic
3. **SIMD**: Manual derivatives can be better vectorized by LLVM
4. **Constant Propagation**: Both benefit equally

### Memory Considerations

✅ **Both achieve zero allocations** - Tensors.jl gradient() is very well optimized for memory

### Decision Tree

**For assembly loops (hot path):**
- ❌ **Do NOT use AD** - 30× overhead is unacceptable
- ✅ **Use hand-coded derivatives** - keep them for Tet10, Hex8, Quad4, Tri3
- ✅ **Verify with AD in unit tests** - catch human errors

**For prototyping/research:**
- ✅ **Use AD freely** - development velocity matters more
- ✅ **Profile before optimizing** - maybe it's not the bottleneck

**For rare elements:**
- ⚠️ **Consider symbolic generation** - SymPy/Symbolics.jl once, use forever
- ✅ **Unit test against AD** - verify correctness

**For exotic bases (NURBS, splines):**
- ✅ **Must use AD** - hand derivatives are intractable
- ⚠️ **Accept performance cost** - no alternative

## Recommendations

### Short Term (Current JuliaFEM)

**Keep manual derivatives for common elements:**
- Tet4, Tet10 (3D volume)
- Hex8, Hex20, Hex27 (3D volume)
- Quad4, Quad8, Quad9 (2D, shells)
- Tri3, Tri6 (2D, shells)
- Seg2, Seg3 (1D, beams)

These elements cover **>95% of real-world usage**. The 30× speedup justifies maintenance.

**Use AD for everything else:**
- Pyramid elements (rare)
- Wedge elements (rare)
- Research elements
- NURBS-based isogeometric analysis

### Long Term (v2.0+)

**Symbolic derivative generation:**
```julia
using Symbolics

# Define basis symbolically once
@variables ξ η ζ
N1_sym = (1 - ξ - η - ζ) * (2*(1 - ξ - η - ζ) - 1)

# Generate Julia code for derivatives
dN1_dξ = Symbolics.derivative(N1_sym, ξ)
code = Symbolics.build_function(dN1_dξ, [ξ, η, ζ])

# Store in basis/generated/Tet10.jl
# Zero human error, zero AD overhead!
```

**Benefits:**
- Hand-level performance
- Zero human errors (symbolic math is exact)
- Easy to add new elements (just define basis symbolically)
- Unit test against AD to verify symbolic engine

## Conclusion

**The data is clear:** For JuliaFEM's performance-critical code (element assembly), **manual derivatives are 30× faster** than AD.

**Recommended strategy:**
1. ✅ Keep hand-coded derivatives for common elements (Tet10, Hex8, Quad4, etc.)
2. ✅ Use AD for prototyping and rare elements
3. ✅ Add unit tests comparing manual vs AD (catch human errors)
4. 🎯 Future: Generate derivatives symbolically (best of both worlds)

**Why not AD everywhere?**
- Assembly loops: millions of evaluations per solve
- 30× overhead = 10+ seconds per solve on realistic problems
- Users will notice the performance difference

**Why not abandon AD?**
- Excellent for prototyping
- Required for exotic bases (NURBS)
- Perfect for unit testing manual derivatives
- Zero allocations makes it usable in inner loops (if needed)

The zero-allocation achievement is impressive, but compute overhead dominates. **Performance-critical code still needs hand-tuned derivatives.**

---

## References

1. ForwardDiff.jl documentation
2. Tensors.jl gradient() implementation
3. "Automatic Differentiation in FEM" - various papers
4. JuliaFEM Issue #XXX: Zero-allocation refactoring

## Appendix: Code Listings

See `benchmarks/tet10_derivatives_benchmark.jl` for full implementations.
