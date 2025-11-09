---
title: "Roadmap to High Performance Computing"
subtitle: "Why we make hard choices and sacrifice convenience for speed"
description: "Technical justification for design decisions prioritizing performance over convenience"
date: 2025-11-09
author: "Jukka Aho"
categories: ["philosophy", "architecture", "performance", "roadmap"]
keywords: ["hpc", "performance", "type stability", "design decisions", "trade-offs"]
audience: "contributors and researchers"
level: "expert"
type: "philosophy"
series: "The JuliaFEM Book"
chapter: "Part III: History and Vision"
status: "living document"
priority: "critical"
---

## Executive Summary

**Goal:** JuliaFEM will be a high-performance contact mechanics library capable of real HPC workloads.

**Philosophy:** Efficiency > Educativeness. When forced to choose, we always choose performance.

**Reality Check:** There is no free lunch. Julia doesn't make miracles. You want speed? Feel the pain.

---

## The Hard Truth About Performance

### Julia is Not Magic

From [Issue #266](https://github.com/JuliaFEM/JuliaFEM.jl/issues/266):

> "Do like Python, be slow like Python. Know what you do before compiling, and be fast like C. There's no free lunch, and Julia is not making miracles in the field of computer science."

Julia solves a problem that shouldn't exist - the same way TypeScript solves a problem that shouldn't exist:

1. **First:** Duck typing! How excellent! Save 4 keystrokes writing `x = 1` instead of `int x = 1`.
2. **Then:** Introduce performance problems and runtime errors.
3. **Then:** Learn TDD, test against errors that wouldn't exist with static types.
4. **Finally:** Get sane. Python adds type hints. JavaScript adds TypeScript.

**Result:** Python is slow, TypeScript is verbose, and we're back to typing. But now with runtime overhead.

### The Field System Trap

**I still believe** everything should be a field. It's elegant, it's flexible, it's FEM-correct.

**But:** `field["foo"] = 2` will NEVER be fast. Not in Python, not in Julia, not anywhere.

```julia
# ❌ Elegant but SLOW (100× slower)
element.fields["displacement"] = u

# ✅ Fast but verbose
get_displacement(element)  # Type-stable dispatch
```

**The Choice:** Speed or convenience. You cannot have both.

**Our Choice:** Speed. Always.

---

## Core Principle: Type Stability Over Everything

### What is Type Stability?

```julia
# ❌ Type-unstable (compiler cannot predict return type)
function foo(flag)
    if flag
        return 1        # Int
    else
        return 1.0      # Float64
    end
end

# ✅ Type-stable (compiler knows return type)
function foo(flag)
    if flag
        return 1.0      # Float64
    else
        return 1.0      # Float64
    end
end
```

**Impact:** Type-unstable code is 10-100× slower. Not 10%, **100× slower**.

### The Dict Problem

```julia
# ❌ Type-unstable Dict (Any values)
fields = Dict{String,Any}()
fields["displacement"] = [1.0, 2.0, 3.0]
u = fields["displacement"]  # Type: Any (compiler lost)

# ✅ Type-stable (but less flexible)
struct Fields{T}
    displacement::Vector{T}
    temperature::Vector{T}
end
```

**Current Status (2015-2019):** JuliaFEM used `Dict{String,Any}` everywhere.

**Result:** 100× performance loss.

**Future:** Replace with type-stable alternatives. No exceptions.

---

## Strategic Decisions and Trade-offs

### Decision 1: No Dynamic Field System

**What we want:**

```julia
element["my_custom_field"] = magic_value  # Beautiful!
```

**Reality:** This is `Dict{String,Any}`. Type-unstable. 100× slower.

**Our Choice:**

```julia
struct Element{N,NIP,M,B}
    # Fixed, type-stable fields only
    connectivity::NTuple{N,UInt}
    integration_points::NTuple{NIP,IP}
end
```

**Justification:** If you want HPC, you sacrifice runtime flexibility for compile-time knowledge.

**When someone asks "Why can't I add custom fields?"**  
→ Point them here. We've thought about this. Performance wins.

---

### Decision 2: Immutable Data Structures

**What's convenient:**

```julia
element.connectivity[1] = 42  # Easy!
```

**What's fast:**

```julia
element = Element(new_connectivity, ...)  # Create new element
```

**Our Choice:** Immutable `struct`, not `mutable struct`.

**Justification:**

- Compiler can optimize better (no aliasing)
- Stack allocation (not heap)
- Thread-safe by default
- Cache-friendly (no pointer chasing)

**Trade-off:** Slightly more verbose code. Worth it for 2-10× speedup.

---

### Decision 3: NTuple Over Vector

**What's convenient:**

```julia
connectivity = [1, 2, 3, 4]  # Dynamic size
push!(connectivity, 5)       # Can grow
```

**What's fast:**

```julia
connectivity = (1, 2, 3, 4)  # Compile-time size
# Cannot grow - that's the point!
```

**Our Choice:** `NTuple{N,UInt}` for connectivity and integration points.

**Justification:**

- Stack allocation (no heap)
- Compile-time size → SIMD optimization
- Zero allocation in loops
- Type-stable iteration

**Trade-off:** Cannot change element topology at runtime. Good! You shouldn't.

---

### Decision 4: Monolithic Over Multi-Package

**What seems good:**

```julia
# Split into 25 packages
using FEMBase, FEMBasis, FEMQuad, FEMSparse, ...
```

**Reality:** Dependency hell, version conflicts, precompilation nightmare.

**Our Choice:** Monolithic core package.

**Justification:** From [TECHNICAL_VISION.md](../../llm/TECHNICAL_VISION.md):
> "Strategic Mistake #1: Multi-package ecosystem. Result: Dependency hell, can't load package."

**Trade-off:** Larger package, longer initial compile. But it works.

---

### Decision 5: Manual Derivatives Over AD (Sometimes)

**Benchmark Results:** [shape_function_derivatives_ad_vs_manual.md](benchmarks/shape_function_derivatives_ad_vs_manual.md)

- Manual: 8.7 ns
- AD (Tensors.jl): 268.1 ns
- **30× difference**

**Our Choice:** Manual derivatives for hot paths (Seg2, Tri3, Quad4, Tet10, Hex8).

**Justification:** Assembly loops call derivatives millions of times. 30× = 10+ seconds per solve.

**Trade-off:** More code to maintain. But we pre-generate it symbolically anyway.

**When to use AD:** Prototyping, rare elements, NURBS. Not in production hot paths.

---

### Decision 6: Matrix-Free Methods Required

**Problem:** Direct solvers don't scale beyond ~100K DOF.

**Reality:** Contact problems = 1M+ DOF. Direct solve = impossible.

**Our Choice:** Design for matrix-free from day 1.

**Justification:** Cannot retrofit efficiently. Data structures must support:

- Element-by-element assembly
- Matrix-vector products without global matrix
- Krylov solvers (GMRES, CG)

**Trade-off:** More complex API. But necessary for HPC.

---

### Decision 7: Explicit Over Implicit

**What seems clever:**

```julia
# Magic happens behind the scenes
problem = Problem(...)
solve(problem)  # What does this do? 🤷
```

**What's maintainable:**

```julia
# Explicit steps
K = assemble_stiffness(problem)
f = assemble_forces(problem)
u = solve_linear_system(K, f)
```

**Our Choice:** Explicit, even if verbose.

**Justification:**

- No hidden state
- No action at distance
- Debuggable
- Teachable

**Trade-off:** More code. But you understand it.

---

## The Hierarchy of Values

When conflicts arise, this is our priority order:

1. **Correctness** - Wrong answers are useless
2. **Performance** - Slow correct answers limit applicability
3. **Maintainability** - Unmaintainable code dies
4. **Educativeness** - Code should teach FEM
5. **Convenience** - Nice to have, never at cost of 1-4

**Note:** Educativeness is #4, not #1. This is intentional.

---

## Specific Technical Requirements

### For Threading (Future)

**Required:**

- Immutable data structures (✅ Element is immutable now)
- No shared mutable state
- Thread-local assembly buffers
- Lock-free data structures

**Implication:** Cannot add thread support later. Design for it now.

---

### For GPU (Future)

**Required:**

- Contiguous memory layouts
- Struct-of-arrays, not array-of-structs
- No dynamic dispatch in kernels
- No allocations in hot loops (✅ tuple returns)

**Implication:** Data layout matters from day 1.

---

### For Distributed (Future)

**Required:**

- Node-based decomposition
- Minimal halo exchange
- Matrix-free operators
- Explicit communication

**Implication:** Cannot bolt MPI onto element-centric design.

---

## What We're Giving Up

Let's be honest about what we're sacrificing:

### 1. Runtime Flexibility

**Gone:** `element["my_field"] = anything`

**Why:** Type stability requires compile-time types.

**Alternative:** Pre-define fields in Element struct. Add new element types if needed.

---

### 2. Dynamic Problem Definition

**Gone:** Modify element connectivity at runtime.

**Why:** `NTuple` is immutable and compile-time sized.

**Alternative:** Build mesh correctly first. Adaptive refinement = new elements, not modified elements.

---

### 3. Duck Typing Convenience

**Gone:** "If it walks like a duck..."

**Why:** Julia's compiler needs concrete types for speed.

**Alternative:** Define interfaces explicitly. Use abstract types correctly.

---

### 4. Small Dependencies

**Gone:** "Import only what you need."

**Why:** Package ecosystem fragmentation killed v0.5.

**Alternative:** Monolithic core. Optional extensions via separate packages.

---

### 5. Beginner-Friendly API

**Gone:** One magic function that does everything.

**Why:** Explicit > Implicit for performance debugging.

**Alternative:** Good documentation. Show the steps.

---

## Success Metrics

We'll know we're on the right path when:

1. **Performance:**
   - ✅ Zero allocations in assembly loops
   - ✅ Type-stable hot paths (@code_warntype clean)
   - 🎯 10× faster than v0.5.1 (single-thread)
   - 🎯 Solves 1M DOF contact problem in < 1 hour
   - 🎯 Scales to 100+ threads
   - 🎯 Runs on GPU

2. **Correctness:**
   - ✅ All tests pass
   - 🎯 Matches Code Aster on verification problems
   - 🎯 Published research using JuliaFEM

3. **Maintainability:**
   - ✅ Clear architecture (layered design)
   - ✅ Good documentation (three-tier manual)
   - 🎯 Contributors can add elements easily
   - 🎯 CI catches regressions

4. **Educativeness:**
   - ✅ Code shows FEM concepts clearly
   - 🎯 The JuliaFEM Book teaches FEM + Julia
   - 🎯 Used in university courses

---

## Common Questions and Answers

### Q: "Why not use Dict for flexibility?"

**A:** We tried. It was 100× slower. [See TECHNICAL_VISION.md, Strategic Mistake #2](../../llm/TECHNICAL_VISION.md).

Performance > Flexibility. Always.

---

### Q: "Why immutable? Mutation is natural for FEM."

**A:**

- Compiler optimizes immutable better (2-10× faster)
- Thread-safe by default (required for parallelism)
- Stack allocation (no GC pauses)

You're confusing physical mutation (displacement changes) with data structure mutation. We update physical state by creating new data structures.

---

### Q: "Why not just use Python/C++ if you want speed?"

**A:**

- Python: 100-1000× slower, requires C extensions
- C++: Fast but no REPL, slow compile, template hell
- Julia: Best of both worlds *if you follow the rules*

The rules: Type stability, immutability, zero allocations. No shortcuts.

---

### Q: "Can't Julia's compiler figure this out?"

**A:** **No.** The compiler is smart but not magic.

```julia
x = Dict{String,Any}()["foo"]  # Compiler: 🤷 no idea what type
```

You must give it type information. Period.

---

### Q: "This seems complicated. Why not keep it simple?"

**A:** Simple (for humans) ≠ Simple (for compiler).

```julia
# Simple for human, nightmare for compiler
field["displacement"] = u

# Verbose for human, trivial for compiler  
get_displacement(element) = element.displacement
```

HPC requires thinking like a compiler, not like a Python programmer.

---

### Q: "When do you compromise?"

**A:** Rarely. Examples:

**OK to compromise:**

- Non-hot paths (I/O, visualization, setup)
- Prototyping phase (before optimization)
- User-facing convenience layers (wrapping fast core)

**Never compromise:**

- Assembly loops
- Solve iterations  
- Inner products
- Jacobian evaluations

If it's called millions of times, it must be perfect.

---

## Roadmap Timeline

### Phase 1: Foundation (✅ Mostly Done)

- ✅ Immutable Element
- ✅ NTuple connectivity/integration points
- ✅ Zero-allocation basis functions
- ✅ Type-stable core types
- ✅ Pre-generated Lagrange bases

### Phase 2: Consolidation (Current - Nov 2025)

- 🔄 Merge 25 vendor packages → monolithic
- 🔄 Remove Dict-based fields
- 🔄 Fix remaining type instabilities
- 🔄 All tests passing

### Phase 3: Performance (Months 5-8)

- 🎯 Benchmark baseline
- 🎯 Profile and optimize hot paths
- 🎯 10× single-thread speedup
- 🎯 Matrix-free assembly

### Phase 4: Parallelism (Months 9-12)

- 🎯 Thread-parallel assembly
- 🎯 GPU kernel prototypes
- 🎯 Distributed mesh partitioning

### Phase 5: Production (Months 12-18)

- 🎯 1M DOF contact problems
- 🎯 100+ thread scaling
- 🎯 Published benchmarks vs commercial codes

---

## How to Use This Document

**When someone asks "Why did you do X?"**

1. Find X in this document
2. Show the justification
3. Link to supporting evidence (benchmarks, old code, theory)

**When you're tempted to add convenience:**

1. Check: Is this in a hot path?
2. If yes: Don't do it. Performance > Convenience.
3. If no: Consider it. Measure first.

**When adding new features:**

1. Is it type-stable? (@code_warntype clean)
2. Zero allocations? (@allocated == 0)
3. Works with threading? (immutable, no shared state)
4. GPU-compatible? (struct-of-arrays, no allocations)

All four must be yes, or you're building technical debt.

---

## References

- [TECHNICAL_VISION.md](../../llm/TECHNICAL_VISION.md) - Strategic mistakes (2015-2019)
- [VISION_2.0.md](../../llm/VISION_2.0.md) - Complete project vision
- [Issue #266: Deal with the type system](https://github.com/JuliaFEM/JuliaFEM.jl/issues/266)
- [Benchmark: AD vs Manual](benchmarks/shape_function_derivatives_ad_vs_manual.md)
- [Lagrange Basis Functions](lagrange_basis_functions.md)

---

## Conclusion

**There is no free lunch.**

HPC requires discipline:

- Type stability (no Any, no Union, no Dict without types)
- Immutability (struct, not mutable struct)
- Zero allocations (tuples, not vectors)
- Explicit (no magic, show the steps)

Is it harder than Python? Yes.  
Is it worth it? **Absolutely.**

10× speedup = 10× larger problems = 10× more science.

**Efficiency > Educativeness.**

We're not building a toy. We're building a tool for real research.

That's the roadmap. That's the commitment.

---

**Status:** Living document, updated as we learn.  
**Last Updated:** November 9, 2025  
**Next Review:** After Phase 2 completion
