---
title: "Test Fixes Needed"
description: "Known test failures and fixes required for full test suite passing"
date: 2025-11-09
updated: 2025-11-09
author: "Jukka Aho"
categories: ["testing", "todo"]
keywords: ["tests", "failures", "fixes", "todo"]
audience: "contributors"
level: "intermediate"
type: "technical note"
series: "Contributor Manual"
status: "active maintenance"
---

# Test Fixes Needed

**Date:** November 8, 2025  
**Status:** 5 passing, 49 failing (infrastructure now in place)

## Summary

Tests are failing due to API evolution between Julia 0.6/1.0 (2018) and Julia 1.12 (2025), not fundamental architectural problems. Package loads successfully and core functionality works.

## Main Issues

### 1. Missing `aster_read_mesh` (14 tests)
**Problem:** Tests use `aster_read_mesh()` from IO submodule, but it requires HDF5  
**Files affected:** Most 3D elasticity tests, med file tests  
**Fix options:**
- A) Add HDF5 as optional dependency (Julia 1.9+ package extensions)
- B) Skip tests that need .med files for now
- C) Convert test meshes to .inp format (ABAQUS, which we support)

**Recommendation:** Option C - convert test meshes to .inp format

### 2. `eval_basis!` Signature Mismatch (2 tests)
**Problem:** `eval_basis!(::Type{Seg2}, ::Matrix, ::Tuple{Float64})`  
**Current:** `eval_basis!(::Seg2, ::Vector, ::Tuple{Float64}, time::Float64)`  
**Location:** `vendor/FEMBasis.jl`

**Fix:** Update signature in FEMBasis or fix call sites

### 3. `jacobian` Signature Mismatch (~20 tests)  
**Problem:** Tests call `jacobian(element_type, X, xi)` with old signatures  
**Current API:** Different parameter order or types  
**Location:** `vendor/FEMBasis.jl/src/jacobian.jl`

**Fix:** Consolidate FEMBasis into src/basis/ with modern API

### 4. `allocate_buffer` Missing (2 tests)
**Problem:** `allocate_buffer(::Problem{Elasticity}, ::Vector{Element})`  
**Status:** Method doesn't exist in current codebase  
**Fix:** Either restore method or update tests to not need it

### 5. `Analysis` Missing (5 tests) - ✅ FIXED
**Status:** Now exported, these tests should pass

### 6. Statistics Package Missing (1 test) - ✅ FIXED  
**Status:** Now in test dependencies

## Test Categories

### ✅ Passing (5 tests)
- Virtual work test
- Contact 2D/3D tests  
- Mortar 2D tests
- Heat transfer (basic)

### ❌ Failing - Missing HDF5 (~14 tests)
- test_elasticity_2d_nonlinear_with_surface_load.jl
- test_elasticity_3d_unit_block.jl
- test_elasticity_med_pyr5_point_load.jl
- test_elasticity_plane_strain.jl
- test_elasticity_pyr5_point_load.jl
- Many more...

### ❌ Failing - API Mismatches (~30 tests)
- eval_basis! signature (2)
- jacobian signature (~20)
- allocate_buffer missing (2)
- Various others (6)

## Action Plan

### Phase 1: Low-Hanging Fruit (1-2 hours)
1. ✅ Export Analysis types
2. ✅ Add Statistics to test deps
3. ⏳ Skip/comment out HDF5-dependent tests temporarily
4. ⏳ Re-run tests, see how many pass

### Phase 2: API Fixes (4-6 hours)  
1. Fix `eval_basis!` signature in FEMBasis
2. Fix `jacobian` signature in FEMBasis
3. Either restore `allocate_buffer` or update tests
4. Fix any remaining signature mismatches

### Phase 3: Mesh Conversion (2-4 hours)
1. Find all .med test meshes
2. Convert to .inp format using Code Aster or similar
3. Update test files to use .inp instead of .med
4. Re-run tests

### Phase 4: Verify All Pass (1 hour)
1. Run full test suite
2. Fix any remaining issues
3. Update CI to run tests automatically
4. Celebrate! 🎉

## Expected Outcome

After these fixes:
- ~40+ tests should pass (out of 56 total)
- CI will catch regressions automatically
- Good foundation for further consolidation work

## Notes

The fact that package loads and 5 tests pass is actually very good news - it means the core architecture is sound. These are just API compatibility issues that accumulated over 6 years of Julia evolution.

Most fixes are mechanical (update signatures) rather than requiring deep understanding of the algorithms.
