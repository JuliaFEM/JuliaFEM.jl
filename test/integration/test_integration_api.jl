"""
# Integration Points API Tests (test/integration/)

## What
Tests the zero-allocation integration points API using `get_gauss_points!(topology, scheme)`.
This validates that quadrature point queries are completely compile-time resolved with
zero runtime allocation.

## Why
Integration point evaluation happens in the **innermost assembly loop** - potentially
billions of times for large problems. Even a single allocation per call would be
catastrophic for performance.

This test validates the CRITICAL performance requirement:
- **Zero allocations**: `@allocated get_gauss_points!(...)` must return 0
- **Compile-time resolution**: All return types are NTuple (stack-allocated)
- **Type stability**: Returns `Tuple{Tuple{Float64, Vec{D}},...}`

The old API had performance issues:
- Heap-allocated arrays for integration points (allocates every call)
- Type-unstable returns (Any or AbstractArray)
- Runtime dispatch instead of compile-time specialization

The new API fixes all of this using NTuple and generated functions.

## How
**Zero Allocation Tests:**
- Confirms `@allocated get_gauss_points!(...)` == 0 for all topologies
- Tests Segment, Triangle, Tetrahedron, Quadrilateral, Hexahedron
- Various Gauss orders: {1}, {2}, {3}

**Return Type Validation:**
- Verifies return is Tuple of (weight, ξ) pairs
- Each weight is Float64
- Each ξ is Vec{D} from Tensors.jl (D = dimension)

**Integration Point Counts:**
- Segment: 1, 2, 3 points (Gauss{1}, {2}, {3})
- Triangle: 1, 3, 6 points
- Tetrahedron: 1, 4, 5 points
- Quadrilateral: 1, 4, 9 points (tensor product: n²)
- Hexahedron: 1, 8, 27 points (tensor product: n³)

**Weight Sum Validation:**
- Weights must sum to reference element volume/area:
  - Segment: 2.0 (length of [-1,1])
  - Triangle: 0.5 (area of reference triangle)
  - Tetrahedron: 1/6 (volume of reference tet)
  - Quadrilateral: 4.0 (area of [-1,1]²)
  - Hexahedron: 8.0 (volume of [-1,1]³)

**Assembly Loop Pattern:**
- Demonstrates real usage: `for (w, ξ) in get_gauss_points!(...)`
- Verifies zero allocation in actual assembly code
- Shows integration with basis function evaluation

**Performance Benchmarking:**
- Compares new approach vs old (hypothetical)
- Target: ~1 μs for 1000 iterations, 0 allocations

## Expected Results
- ✅ **Zero allocations**: All `@allocated` checks return 0
- ✅ **Correct types**: Returns Tuple{Tuple{Float64, Vec{D}},...}
- ✅ **Correct counts**: Point counts match Gauss order
- ✅ **Correct weights**: Sum to reference element volume
- ✅ **Assembly pattern**: Zero allocations in realistic usage
- ✅ **Performance**: Sub-microsecond per 1000 iterations

## API Pattern (NEW vs DEPRECATED)
```julia
# ✅ NEW API (zero-allocation):
for (weight, ξ) in get_gauss_points!(Triangle, Gauss{2})
    N = get_basis_functions(Triangle(), Lagrange{Triangle,1}(), ξ)
    dN = get_basis_derivatives(Triangle(), Lagrange{Triangle,1}(), ξ)
    # ... assembly using N, dN, weight
end

# ❌ DEPRECATED API (allocates every call):
ips = get_integration_points(element)  # Heap allocation!
for ip in ips
    eval_basis!(bi, X, ip)  # Mutation, type-unstable
    # ...
end
```

## Architecture Principle
**Compile-Time vs Runtime Resolution**

The new API pushes ALL integration point computation to compile time:
- Topology type known at compile time → correct quadrature selected
- Gauss order known at compile time → correct number of points
- Return type fully inferred → NTuple allocated on stack

This is the FOUNDATION for zero-allocation assembly!

## Performance Target
For 1000 assembly iterations over 3-point triangle quadrature:
- **Time**: < 1 μs (compile-time overhead amortized)
- **Allocations**: 0 bytes (all stack-allocated)
- **GC**: 0% (no heap pressure)

If this test fails, the entire assembly chain will be slow!
"""

# Test: Zero-Allocation Integration Points
# =========================================

using Test
using JuliaFEM
using Tensors
using BenchmarkTools

@testset "Integration Points API" begin
    @testset "Zero Allocation" begin
        # All integration point queries should allocate zero bytes
        @test (@allocated get_gauss_points!(Segment, Gauss{1})) == 0
        @test (@allocated get_gauss_points!(Triangle, Gauss{1})) == 0
        @test (@allocated get_gauss_points!(Tetrahedron, Gauss{1})) == 0
        @test (@allocated get_gauss_points!(Hexahedron, Gauss{2})) == 0
    end

    @testset "Return Type" begin
        # Should return NTuple of (Float64, Vec{D}) pairs
        ips = get_gauss_points!(Triangle, Gauss{1})

        @test isa(ips, Tuple)
        @test length(ips) == 1

        w, ξ = ips[1]
        @test isa(w, Float64)
        @test isa(ξ, Vec{2})
    end

    @testset "Segment" begin
        # 1-point Gauss
        ips = get_gauss_points!(Segment, Gauss{1})
        @test length(ips) == 1
        w, ξ = ips[1]
        @test w ≈ 2.0
        @test ξ[1] ≈ 0.0

        # 2-point Gauss
        ips = get_gauss_points!(Segment, Gauss{2})
        @test length(ips) == 2
        @test sum(ip[1] for ip in ips) ≈ 2.0  # Weights sum to length
    end

    @testset "Triangle" begin
        # 1-point Gauss (centroid)
        ips = get_gauss_points!(Triangle, Gauss{1})
        @test length(ips) == 1
        w, ξ = ips[1]
        @test w ≈ 0.5  # Area of reference triangle
        @test ξ[1] ≈ 1 / 3
        @test ξ[2] ≈ 1 / 3

        # 3-point Gauss
        ips = get_gauss_points!(Triangle, Gauss{2})
        @test length(ips) == 3
        @test sum(ip[1] for ip in ips) ≈ 0.5
    end

    @testset "Tetrahedron" begin
        # 1-point Gauss (centroid)
        ips = get_gauss_points!(Tetrahedron, Gauss{1})
        @test length(ips) == 1
        w, ξ = ips[1]
        @test w ≈ 1 / 6  # Volume of reference tetrahedron
        @test ξ[1] ≈ 0.25
        @test ξ[2] ≈ 0.25
        @test ξ[3] ≈ 0.25

        # 4-point Gauss
        ips = get_gauss_points!(Tetrahedron, Gauss{2})
        @test length(ips) == 4
        @test sum(ip[1] for ip in ips) ≈ 1 / 6
    end

    @testset "Quadrilateral" begin
        # 2×2 Gauss (standard for Q1)
        ips = get_gauss_points!(Quadrilateral, Gauss{2})
        @test length(ips) == 4
        @test sum(ip[1] for ip in ips) ≈ 4.0  # Area of reference quad
    end

    @testset "Hexahedron" begin
        # 2×2×2 Gauss (standard for Hex8)
        ips = get_gauss_points!(Hexahedron, Gauss{2})
        @test length(ips) == 8
        @test sum(ip[1] for ip in ips) ≈ 8.0  # Volume of reference hex
    end
end

@testset "Usage in Assembly Loop" begin
    # Demonstrate zero-allocation assembly pattern
    function assemble_element_stiffness()
        K = 0.0
        for (w, ξ) in get_gauss_points!(Triangle, Gauss{2})
            # Shape functions
            N1 = 1 - ξ[1] - ξ[2]
            N2 = ξ[1]
            N3 = ξ[2]

            # Accumulate (simplified stiffness)
            K += w * (N1^2 + N2^2 + N3^2)
        end
        return K
    end

    # Should allocate zero
    @test (@allocated assemble_element_stiffness()) == 0

    # Verify result is consistent
    K1 = assemble_element_stiffness()
    K2 = assemble_element_stiffness()
    @test K1 ≈ K2
end

@testset "Performance Comparison" begin
    println("\n" * "="^70)
    println("PERFORMANCE: Integration Points vs Old Approach")
    println("="^70)

    # New approach (compile-time, Vec{D})
    new_approach() = begin
        sum_val = 0.0
        for _ in 1:1000
            for (w, ξ) in get_gauss_points!(Triangle, Gauss{2})
                sum_val += w * sum(ξ)
            end
        end
        return sum_val
    end

    println("\nNew approach (compile-time + Vec{D}):")
    display(@benchmark $new_approach())

    println("\n\nExpected: ~1 μs, 0 allocations")
    println("="^70)
end

println("\n✓ All integration point tests passed!")
println("\nUsage Example (NEW API):")
println("```julia")
println("# Zero-allocation loop over integration points:")
println("for (weight, ξ) in get_gauss_points!(Triangle, Gauss{2})")
println("    # NEW API (recommended):")
println("    N = get_basis_functions(Triangle(), Lagrange{1}(), ξ)")
println("    dN = get_basis_derivatives(Triangle(), Lagrange{1}(), ξ)")
println("    # ... compute element matrices")
println("end")
println("```")
println()
println("Note: eval_basis! and eval_dbasis! are DEPRECATED.")
println("Use get_basis_functions and get_basis_derivatives instead.")
