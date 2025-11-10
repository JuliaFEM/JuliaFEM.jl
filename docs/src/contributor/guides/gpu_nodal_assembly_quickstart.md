---
title: "GPU Nodal Assembly - Quick Start Guide"
date: 2025-11-10
author: "JuliaFEM Team"
status: "Authoritative"
last_updated: 2025-11-10
tags: ["gpu", "nodal-assembly", "nonlinear", "quickstart", "guide"]
---

**Status:** CPU ✅ Working | GPU 🔄 Ready to Test

---

## What is This?

A **complete GPU-resident nonlinear FEM solver** using:
- **Nodal assembly** (matrix-free, no atomics)
- **Tensors.jl** (natural tensor operations on GPU)
- **Two-phase pipeline** (GP data → nodal assembly)
- **Perfect plasticity** (von Mises with return mapping)

---

## Quick Test (CPU Reference)

```bash
cd /home/juajukka/dev/JuliaFEM.jl
julia demos/nodal_assembly_cpu.jl
```

**Expected output:**
```
Residual norm: 727.2081516082287
Material States: Plastic (α = 5.634921e-03) at all GPs
Force Balance: ✅ PASSED
```

---

## Quick Test (GPU)

```bash
cd /home/juajukka/dev/JuliaFEM.jl
julia --project=. demos/nodal_assembly_gpu.jl
```

**Requirements:**
- CUDA-capable GPU
- CUDA.jl installed

**Expected output:**
- Residual norm should match CPU: ~727.2
- All material states plastic
- Force balance passed

---

## Architecture Overview

### Two-Phase Pipeline

```
Phase 1: Integration Point Data (GP Kernel)
  Input:  u, nodes, elements, states_old
  Output: σ_gp (stresses), states_new
  Parallelism: One thread per GP
  
  ↓

Phase 2: Nodal Assembly (Node Kernel)
  Input:  σ_gp, nodes, elements, node_to_elems (CSR)
  Output: r (residual vector)
  Parallelism: One thread per node
  NO ATOMICS NEEDED!
```

### Key Data Structures

```julia
# Stresses (Tensors.jl on GPU!)
σ_gp = CuArray{SymmetricTensor{2,3,Float64,6}, 1}

# Material states
states = CuArray{PlasticState, 1}

# CSR map (which elements touch each node)
struct NodeToElementsMap
    ptr::CuArray{Int32, 1}
    data::CuArray{Int32, 1}
end
```

---

## Files to Know

### Documentation
- **`docs/design/gpu_nodal_assembly_architecture.md`** - Complete architecture (500+ lines)
- **`llm/sessions/2025-11-10_gpu_nodal_assembly_complete.md`** - Session summary

### Implementation
- **`demos/nodal_assembly_cpu.jl`** - CPU reference (400+ lines, ✅ working)
- **`demos/nodal_assembly_gpu.jl`** - GPU version (450+ lines, ready to test)

### Background
- **`demos/newton_krylov_anderson_cpu.jl`** - Complete Newton-Krylov solver
- **`docs/design/gpu_solver_strategy_expert_validated.md`** - Expert-validated strategy

---

## Why Nodal Assembly?

### ❌ Element-Based (Standard GPU FEM)
```julia
for elem in elements
    compute element forces
    CUDA.@atomic r[node] += f_elem[i]  # ATOMIC - CONTENTION!
end
```

### ✅ Node-Based (Our Approach)
```julia
for node in nodes  # Each thread owns ONE node
    for elem in elements_touching_node
        f_node += contribution from elem
    end
    r[node] = f_node  # DIRECT WRITE - NO ATOMICS!
end
```

**Benefits:**
- No atomic operations (faster!)
- Matrix-free (lower memory)
- Contact-ready (contact is nodal)
- Scalable (perfect parallelism)

---

## Next Steps (Prioritized)

### 1. Test GPU Implementation 🔄 IMMEDIATE
```bash
julia --project=. demos/nodal_assembly_gpu.jl
```
Verify results match CPU reference.

### 2. Add Line Search ⚠️ CRITICAL
Current Newton solver diverges. Need backtracking line search.

### 3. Add Preconditioning 🎯 PERFORMANCE
Chebyshev-Jacobi → GMG. Expert says: "THE critical factor."

### 4. Integrate with Newton-Krylov
Replace element assembly with GPU nodal assembly.

---

## Performance Expectations

### Phase 1 (GP Data)
- **Compute-bound** (plasticity return mapping)
- 1M GPs: ~10-100ms on modern GPU
- Scales linearly with GP count

### Phase 2 (Nodal Assembly)
- **Memory-bound** (CSR traversal, stress reads)
- 100K nodes: ~5-50ms on modern GPU
- Depends on node connectivity

### Overall
- Small meshes (<1K elements): GPU overhead dominates
- Medium meshes (~10K elements): Breakeven point
- Large meshes (100K+ elements): 10-100× speedup expected

---

## Troubleshooting

### GPU Kernel Doesn't Compile
- Check CUDA.jl is installed: `using CUDA; CUDA.functional()`
- Check Tensors.jl version compatible with CUDA.jl
- Simplify kernel (remove plasticity, test with elastic only)

### Results Don't Match CPU
- Check thread indexing (1-based in Julia!)
- Check CSR map built correctly
- Compare GP-by-GP (print intermediate values)

### Force Balance Fails
- Check Gauss weights (should sum to element volume)
- Check detJ computation (should be positive)
- Check node ordering (right-hand rule)

---

## The Grand Vision

**Goal:** Complete GPU-resident nonlinear FEM solver

**Pipeline:**
```
Augmented Lagrangian (for contact)
  ↓ Anderson acceleration HERE
Newton Loop
  ↓ Line search for globalization
GMRES (preconditioned)
  ↓ GMG preconditioner
  ↓ Eisenstat-Walker forcing
Matrix-vector product:
  ↓ Phase 1: compute_gp_data_kernel!()
  ↓ Phase 2: nodal_assembly_kernel!()
  
ALL ON GPU - NO CPU TRANSFERS!
```

---

## Success Criteria

### ✅ Achieved (November 10, 2025)
- [x] Architecture documented
- [x] CPU reference working
- [x] GPU implementation complete
- [x] Tensors.jl validated

### 🔄 Next Session
- [ ] GPU kernels tested on hardware
- [ ] Results match CPU reference
- [ ] Force balance passes on GPU

### 🎯 Near-Term Goals
- [ ] Newton solver converges
- [ ] GMRES preconditioned
- [ ] GPU-resident solver working

---

## Quick Reference

**Test CPU:**
```bash
julia demos/nodal_assembly_cpu.jl
```

**Test GPU:**
```bash
julia --project=. demos/nodal_assembly_gpu.jl
```

**Check architecture:**
```bash
cat docs/design/gpu_nodal_assembly_architecture.md
```

**Check session notes:**
```bash
cat llm/sessions/2025-11-10_gpu_nodal_assembly_complete.md
```

---

**Ready to test the beast! 🚀**
