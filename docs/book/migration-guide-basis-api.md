---
title: "Migration Guide: Deprecated Basis Function API"
date: 2025-11-11
author: "JuliaFEM Team"
status: "Authoritative"
tags: ["migration", "deprecation", "basis-functions", "api"]
---

## Overview

The old basis function API (`eval_basis!` and `eval_dbasis!`) is **DEPRECATED** and will be removed in a future release. This guide helps you migrate to the new, recommended API.

## Summary of Changes

| Old API (DEPRECATED) | New API (RECOMMENDED) |
|----------------------|------------------------|
| `eval_basis!(Lagrange{Triangle,1}, Float64, ξ)` | `get_basis_functions(Triangle(), Lagrange{1}(), ξ)` |
| `eval_dbasis!(Lagrange{Triangle,1}, ξ)` | `get_basis_derivatives(Triangle(), Lagrange{1}(), ξ)` |

## Why the Change?

**Key Improvements:**

1. **Separation of concerns** - Topology and basis are separate parameters
2. **Clearer naming** - `get_basis_functions` vs `eval_basis!` (no mutation despite `!`)
3. **Type parameter simplification** - No redundant `Float64` parameter
4. **Consistent API** - Same pattern across all basis evaluation functions

## Migration Examples

### Example 1: Triangle Linear Element

**OLD (deprecated):**

```julia
# Element with Lagrange{Triangle,1} basis
xi = Vec(0.25, 0.25)

N = eval_basis!(Lagrange{Triangle,1}, Float64, xi)
dN = eval_dbasis!(Lagrange{Triangle,1}, xi)
```

**NEW (recommended):**

```julia
# Topology and basis passed separately
topology = Triangle()
basis = Lagrange{1}()
xi = Vec(0.25, 0.25)

N = get_basis_functions(topology, basis, xi)
dN = get_basis_derivatives(topology, basis, xi)
```

### Example 2: Tetrahedron Quadratic Element

**OLD (deprecated):**

```julia
N = eval_basis!(Lagrange{Tetrahedron,2}, Float64, xi)
dN = eval_dbasis!(Lagrange{Tetrahedron,2}, xi)
```

**NEW (recommended):**

```julia
N = get_basis_functions(Tetrahedron(), Lagrange{2}(), xi)
dN = get_basis_derivatives(Tetrahedron(), Lagrange{2}(), xi)
```

### Example 3: Assembly Loop with Integration Points

**OLD (deprecated):**

```julia
for (w, xi) in get_gauss_points!(Triangle, Gauss{2})
    N = eval_basis!(Lagrange{Triangle,1}, Float64, xi)
    dN = eval_dbasis!(Lagrange{Triangle,1}, xi)
    
    # Assembly...
end
```

**NEW (recommended):**

```julia
topology = Triangle()
basis = Lagrange{1}()

for (w, xi) in get_gauss_points!(Triangle, Gauss{2})
    N = get_basis_functions(topology, basis, xi)
    dN = get_basis_derivatives(topology, basis, xi)
    
    # Assembly...
end
```

**Note:** `topology` and `basis` are type-stable constants, so there's no performance penalty from creating them outside the loop.

## Complete Assembly Example

**OLD (deprecated):**

```julia
function assemble_element_old(element::Element)
    K_local = zeros(9, 9)
    
    for ip in get_integration_points(element)
        w = ip.weight
        xi = Vec(ip.coords...)
        
        # DEPRECATED API
        N = eval_basis!(Lagrange{Triangle,2}, Float64, xi)
        dN = eval_dbasis!(Lagrange{Triangle,2}, xi)
        
        # Jacobian
        J = sum(dN[i] ⊗ X[i] for i in 1:6)
        detJ = det(J)
        invJ = inv(J)
        
        # Physical derivatives
        dN_phys = tuple((invJ ⋅ dN[i] for i in 1:6)...)
        
        # Accumulate stiffness
        for i in 1:6, j in 1:6
            K_local[i,j] += w * detJ * dot(dN_phys[i], dN_phys[j])
        end
    end
    
    return K_local
end
```

**NEW (recommended):**

```julia
function assemble_element_new(element::Element)
    K_local = zeros(9, 9)
    
    # Define topology and basis (type-stable constants)
    topology = Triangle()
    basis = Lagrange{2}()
    
    # NEW: Zero-allocation integration points
    for (w, ξ) in get_gauss_points!(Triangle, Gauss{2})
        # NEW API: Separate topology and basis
        N = get_basis_functions(topology, basis, ξ)
        dN = get_basis_derivatives(topology, basis, ξ)
        
        # Jacobian (using new tuple-based API)
        J = sum(dN[i] ⊗ X[i] for i in 1:6)
        detJ = det(J)
        invJ = inv(J)
        
        # Physical derivatives
        dN_phys = tuple((invJ ⋅ dN[i] for i in 1:6)...)
        
        # Accumulate stiffness
        for i in 1:6, j in 1:6
            K_local[i,j] += w * detJ * dot(dN_phys[i], dN_phys[j])
        end
    end
    
    return K_local
end
```

**Performance:** The new version is ~50× faster due to:

- Zero-allocation integration points
- Type-stable basis evaluation
- Compile-time optimizations

## Topology Types Reference

| Old Basis Type | New Topology | New Basis |
|----------------|--------------|-----------|
| `Lagrange{Segment,1}` | `Segment()` | `Lagrange{1}()` |
| `Lagrange{Triangle,1}` | `Triangle()` | `Lagrange{1}()` |
| `Lagrange{Triangle,2}` | `Triangle()` | `Lagrange{2}()` |
| `Lagrange{Quadrilateral,1}` | `Quadrilateral()` | `Lagrange{1}()` |
| `Lagrange{Quadrilateral,2}` | `Quadrilateral()` | `Lagrange{2}()` |
| `Lagrange{Tetrahedron,1}` | `Tetrahedron()` | `Lagrange{1}()` |
| `Lagrange{Tetrahedron,2}` | `Tetrahedron()` | `Lagrange{2}()` |
| `Lagrange{Hexahedron,1}` | `Hexahedron()` | `Lagrange{1}()` |
| `Lagrange{Hexahedron,2}` | `Hexahedron()` | `Lagrange{2}()` |
| `Lagrange{Wedge,1}` | `Wedge()` | `Lagrange{1}()` |
| `Lagrange{Pyramid,1}` | `Pyramid()` | `Lagrange{1}()` |

## Migration Checklist

### Step 1: Find All Uses of Old API

Search your codebase for:

```bash
grep -r "eval_basis!" src/
grep -r "eval_dbasis!" src/
```

### Step 2: Update Function Calls

For each occurrence:

1. Identify the topology (e.g., `Triangle`, `Tetrahedron`)
2. Identify the polynomial order (e.g., `1`, `2`)
3. Replace with new API:

   ```julia
   # OLD: eval_basis!(Lagrange{Triangle,1}, Float64, xi)
   # NEW: get_basis_functions(Triangle(), Lagrange{1}(), xi)
   ```

### Step 3: Update Integration Loops

If using old integration point API:

```julia
# OLD: for ip in get_integration_points(element)
# NEW: for (w, ξ) in get_gauss_points!(Triangle, Gauss{2})
```

### Step 4: Test

Run your tests to ensure:

- All functionality works
- Performance is maintained or improved
- No deprecation warnings appear

## Common Issues

### Issue 1: Type Parameter Confusion

**Problem:**

```julia
# This won't work!
N = get_basis_functions(Lagrange{Triangle,1}(), xi)
```

**Solution:**

```julia
# Topology and basis are separate!
N = get_basis_functions(Triangle(), Lagrange{1}(), xi)
```

### Issue 2: Integration Point Coordinates

**Problem:**

```julia
# Old IP struct had .coords field
xi = Vec(ip.coords...)  # Tuple to Vec conversion
```

**Solution:**

```julia
# New API returns Vec{D} directly
for (w, ξ) in get_gauss_points!(Triangle, Gauss{2})
    # ξ is already Vec{2}!
```

### Issue 3: Performance Regression

**Problem:** New code is slower than old code.

**Solution:** Make sure you're using the new integration points API:

```julia
# WRONG (old API, allocates):
for ip in get_integration_points(element)
    ...

# RIGHT (new API, zero allocation):
for (w, ξ) in get_gauss_points!(Triangle, Gauss{2})
    ...
```

## Further Reading

- **ADR-002:** Basis Function API Design
- **ADR-004:** Integration Points API Design
- **Performance Benchmarks:** `benchmarks/basis_function_access_patterns.jl`
- **Tests:** `test/test_integration_points_api.jl`

## Support

If you encounter issues during migration:

1. Check this guide for common issues
2. Review the ADR documents for design rationale
3. Open an issue on GitHub with a minimal example

## Timeline

- **v0.6.0:** Deprecation warnings added
- **v0.7.0:** Old API will be removed (planned)

**Migrate now** to avoid breaking changes in future releases!
