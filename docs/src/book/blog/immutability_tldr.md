---
title: "Copying is Faster Than Mutating: A Counterintuitive Performance Win"
author: "Jukka Aho"
date: "2025-11-09"
categories: ["Performance", "Benchmarks"]
tags: ["immutability", "type-stability", "quick-reference"]
description: "TL;DR version: How immutable elements are 130x faster with zero allocations"
---

## TL;DR

We made our FEM code **130x faster** by making it immutable. Yes, copying everything is faster than mutating in place. No, we're not crazy. We have benchmarks.

## The "Stupid" Idea

**Old code (mutable Dict):**

```julia
element.fields[:E] = 210e9  # Mutate in place - "fast"
```

**New code (immutable NamedTuple):**

```julia
element = update(element, E=210e9)  # Copy entire struct - "slow"
```

Which do you think is faster?

## The Shocking Results

```text
Field access:         41x faster (immutable)
Assembly loop:       130x faster (immutable)  
1000 element mesh:   120x faster (immutable)

Memory allocations:  70,000 → 0 (immutable)
```

**Immutable is 100x faster AND uses zero memory.**

## The Secret: Type Stability

```julia
# Dict{Symbol,Any} - Type unstable
element.fields[:E]  # Compiler: "What type is this? 🤷"
# Cost: hash lookup + pointer chase + runtime dispatch ≈ 45 nanoseconds

# NamedTuple{(:E,:ν),Tuple{Float64,Float64}} - Type stable  
element.fields.E    # Compiler: "Float64 at offset 0. Got it."
# Cost: inline to CPU register ≈ 1 nanosecond
```

**45x slower just to read a field.** Multiply by millions of accesses in FEM assembly.

## The Compiler Magic

When you write:

```julia
element = ImmutableElement((1,2,3,4), (E=210e9, ν=0.3))
element = update(element, temperature=293.15)
```

The compiler sees:

- Old element not used → reuse stack space
- New element same size → copy is one assignment
- All types known → inline everything
- Result: **Zero heap allocations, SIMD vectorization, GPU-ready**

When you write:

```julia
element.fields[:temperature] = 293.15
```

The Dict must:

- Compute hash of `:temperature`
- Check if key exists (pointer chasing)
- Maybe resize Dict (heap allocation)
- Store as `Any` → runtime dispatch on next access
- Result: **Heap allocations, type instability, CPU-only**

## Real-World Impact

**Assemble 10,000 element mesh:**

| Implementation | Time | Memory | GPU |
|----------------|------|---------|-----|
| Dict (mutable) | 2.4s | 450 MB, 7M allocs | ✗ |
| NamedTuple (immutable) | **0.02s** | **0 MB, 0 allocs** | ✓ |

Interactive vs coffee break. Million-element mesh vs out-of-memory. GPU vs CPU-only.

## The Lesson

Your programming intuition is from 1990s C/C++:

- ✓ Mutation is fast ← **TRUE IN C**
- ✓ Copying is slow ← **TRUE IN C**
- ✗ Type doesn't matter ← **FALSE IN MODERN COMPILERS**

2025 reality:

- **Type stability is everything**
- Compiler optimizes away struct copies
- Mutation breaks type inference
- Immutability enables GPU acceleration

## Try It Yourself

```bash
git clone https://github.com/JuliaFEM/JuliaFEM.jl
cd JuliaFEM.jl
julia benchmarks/element_immutability_benchmark.jl
```

Full article: `docs/blog/immutability_performance.md`

## Bottom Line

We made the "wrong" choice (copy everything, mutate nothing) and got:

- 130x faster code
- Zero allocations  
- GPU compatibility
- Better parallelization

**Copying > Mutating. Immutability > Mutation. Type stability > Everything.**

Measure, don't assume. The evidence is in the benchmarks.

---

*JuliaFEM 1.0 architecture, November 2025*  
*Benchmark: Intel i7-12700K, Julia 1.12.1*  
*Full results in repository*
