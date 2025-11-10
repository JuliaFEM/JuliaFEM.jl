---
title: "GPU Solver Strategy (Expert-Validated)"
date: 2025-11-10
author: "Jukka Aho"
status: "Authoritative"
tags: ["gpu", "solver", "expert-validated", "nonlinear", "contact"]
---

**Based on:** Expert feedback from high-performance nonlinear solid/contact community

## Executive Summary

Our matrix-free Newton-Krylov approach is **correct and industry-standard** for
GPU-based nonlinear FEM. The expert validates our core direction but identifies
**preconditioning as the critical success factor** we haven't fully addressed
yet.

**Key insight:** Anderson acceleration belongs on **outer fixed-point loops**
(ALM, PTC), NOT as a Newton replacement. Invest heavily in **GPU-friendly
preconditioners**.

---

## 1. Nonlinear Strategy (Outer Level)

### ✅ What We're Doing Right

- Matrix-free Newton-Krylov (correct!)
- Planning for Anderson acceleration (correct placement needed - see below)

### 🔧 What to Add

#### 1.1 Globalization (CRITICAL - Currently Missing)

```julia
# CURRENT (naive):
u_new = u + du

# NEEDED (with line search):
α = backtracking_line_search(u, du, r_norm)
u_new = u + α * du
```

**Implementation:**

- Backtracking line search (simple, robust)
- Trust region method (more sophisticated, optional)
- **Start with backtracking** - it's proven and GPU-friendly

#### 1.2 Eisenstat-Walker Forcing Terms (Adaptive Tolerance)

```julia
# CURRENT (fixed GMRES tolerance):
gmres!(du, J, -r, tol=1e-6)

# NEEDED (adaptive):
η = min(0.5, sqrt(||r_k|| / ||r_{k-1}||))  # Eisenstat-Walker formula
gmres!(du, J, -r, tol=η * ||r||)
```

**Benefit:** Reduces wasted GMRES iterations when far from solution.

#### 1.3 Anderson Placement (CORRECTED)

**❌ WRONG (our current plan):**

```julia
# Wrapping Newton iterations
for iter in 1:max_iter
    du = gmres_solve(J, -r)
    u_new = u + du
    u = anderson_step(u_new)  # ← Not effective here!
end
```

**✅ RIGHT (expert recommendation):**

**Use Case 1**: Augmented Lagrangian (ALM) for Contact

```julia
# Outer ALM loop
for alm_iter in 1:max_alm
    # Solve augmented problem with Newton-Krylov
    u = newton_solve(u, λ, ε)
    
    # Update multipliers
    λ_new = λ + ε * g(u)
    
    # Anderson acceleration HERE!
    λ = anderson_step(λ_new, g(u))
end
```

**Use Case 2**: Pseudo-Transient Continuation (PTC)

```julia
# Outer time-stepping loop
for step in 1:max_steps
    # Solve pseudo-time step
    u_new = newton_solve(u_old, Δt)
    
    # Anderson acceleration HERE!
    u = anderson_step(u_new, residual)
    
    # Adapt time step
    Δt = adapt_timestep(convergence_rate)
end
```

**Bottom line:** Anderson accelerates **fixed-point outer loops**, not Newton itself!

---

## 2. Preconditioning (THE CRITICAL PIECE)

### 2.1 Why ILU/ICC Doesn't Work on GPU

**Problem:** ILU(k) requires:

- Sequential triangular solves
- Irregular sparsity patterns
- Fine-grained dependencies

**Result:** 10-100× slower on GPU than CPU!

### 2.2 GPU-Friendly Preconditioners

#### Option 1: p-then-h Geometric Multigrid (GMG) ⭐ RECOMMENDED

**The Strategy:**

```text
Fine level:   P2 (quadratic) operator (matrix-free)
              ↓ p-coarsening
Coarse level: P1 (linear) operator (matrix-free or assembled)
              ↓ h-coarsening (if mesh hierarchy available)
              ↓ h-coarsening
Coarsest:     AMG or BDDC solve (CPU is OK)
```

**Smoothers (GPU-optimized):**

1. **Chebyshev-accelerated Jacobi** (simplest, very effective)

   ```julia
   # Only needs diagonal!
   D = diag(K)
   
   # Chebyshev iteration (no matrix storage)
   for i in 1:num_smoothing_steps
       r = b - K*x  # Matrix-free!
       x += polynomial_weight[i] * (D \ r)
   end
   ```

2. **Element-Block Jacobi**

   ```julia
   # Per-element dense solve (batched on GPU)
   for elem in elements
       K_elem = assemble_element_stiffness(elem)
       x_elem = K_elem \ r_elem  # Small dense solve (4×4 for Tet4)
   end
   ```

3. **Vertex-Star Additive Schwarz** (strongest, more complex)

   ```julia
   # Per-vertex patch solve
   for vertex in vertices
       star = elements_touching(vertex)
       K_patch = assemble_patch(star)  # ~20×20 for Tet4
       x_patch = K_patch \ r_patch     # Batched dense solve
   end
   ```

**Implementation Priority:**

1. Start with Chebyshev-Jacobi (easiest)
2. Add element-block Jacobi (moderate effort)
3. Try vertex-star Schwarz (highest payoff, most complex)

#### Option 2: Two-Level Schwarz + Coarse (When h-hierarchy is messy)

**Structure:**

```text
Fine:   Overlapping additive Schwarz (vertex/edge patches)
Coarse: BDDC or FETI-DP (subdomain-based)
```

**When to use:** Unstructured/distorted meshes where GMG struggles.

#### Option 3: Nonlinear Preconditioning (For Contact/Plasticity)

**ASPIN / RASPEN approach:**

```julia
function nonlinear_preconditioner(u, r)
    # Apply local Newton solves per patch
    for patch in patches
        u_patch_new = local_newton_solve(u_patch, r_patch)
        u_patch = u_patch_new
    end
    return u
end

# Then use in global Newton:
for iter in 1:max_iter
    r = residual(u)
    u_tilde = nonlinear_preconditioner(u, r)  # ← Before linear solve
    du = gmres_solve(J, -r, precond=M)
    u = u + du
end
```

**Benefit:** Handles localized plasticity/contact much better than linear preconditioners.

---

## 3. Contact Formulation

### Current Plan: Mortar Contact (Good!)

### Recommended: Augmented Lagrangian (ALM) ⭐

**Why:**

- Keeps penetration small without penalty's ill-conditioning
- Outer ALM loop = perfect place for Anderson acceleration
- Robust and proven

**Structure:**

```julia
# ALM outer loop (Anderson-accelerated)
λ = zeros(n_contact_dofs)  # Lagrange multipliers
ε = penalty_parameter

for alm_iter in 1:max_alm
    # Augmented problem: min L(u,λ) + ε/2 ||g(u)||²
    u = newton_krylov_solve(u, λ, ε)
    
    # Multiplier update
    g = penetration(u)  # Gap function
    λ_new = λ + ε * g
    
    # Anderson acceleration
    λ = anderson_step(λ_new, g)
    
    # Check convergence
    if norm(g) < tol && norm(λ_new - λ) < tol
        break
    end
end
```

### Alternative: Semi-Smooth Newton (For Experts)

**When:** Contact sets flip frequently, ALM struggles.

**Approach:** Primal-dual active set method with KKT system.

**Preconditioner:** Block preconditioner

```text
[K   B^T] [Δu]   [r_u]
[B   -εI] [Δλ] = [r_λ]

Preconditioner: M ≈ [K^{-1}    0  ]
                    [0      M_λ^{-1}]

```text
where `K^{-1} ≈ GMG` and `M_λ ≈ mass matrix`.

---

## 4. Material Nonlinearity

### 4.1 Consistent Tangents (CRITICAL)

**Current (JFNK - Jacobian-Free):**
```julia
Jv ≈ (R(u + ε*v) - R(u)) / ε  # Finite difference
```

**Problem:** More Newton iterations, sensitive to ε choice.

**Solution:** Provide **consistent algorithmic tangent**:

```julia
# In return mapping, also compute:
dσ/dε = ∂σ/∂ε_total  # Consistent tangent modulus

# Then use in Jacobian application (still matrix-free!)
```

**Implementation plan:**

1. Keep JFNK for prototype (works, simpler)
2. Add consistent tangents for production (fewer iterations)

### 4.2 Near-Incompressibility (ν → 0.5)

**Problem:** Volumetric locking, poor conditioning.

**Solution 1: F-bar Method** (Simpler)

```julia
# Modify deformation gradient to be volume-preserving
F_bar = (det(F_avg))^(1/3) * F * (det(F))^(-1/3)
```

**Solution 2: Mixed u-p Formulation** (Robust but complex)

```julia
# Separate displacement and pressure
minimize L(u, p) subject to div(u) - p/K = 0

# Leads to saddle-point system
# Need block preconditioner (see §3 contact)
```

**Recommendation:** Start with F-bar (easier), use u-p if needed.

---

## 5. GPU Implementation Strategy

### 5.1 Kernel Design for P2 Tetrahedra

**Challenge:** No tensor-product structure (unlike hexes/quads).

**Optimization strategies:**

1. **Warp-per-element** (32 threads collaborate)

   ```julia
   @cuda threads=32 blocks=n_elements÷32 kernel(...)
   
   # Inside kernel:
   warp_id = threadIdx().x
   if warp_id <= 4  # 4 Gauss points for Tet10
       # Each thread handles one integration point
       # Warp reduction before atomic scatter
   end
   ```

2. **Structure-of-Arrays (SoA) layout**

   ```julia
   # BAD (Array-of-Structs):
   nodes = [Node(x,y,z) for _ in 1:n]  # Strided access
   
   # GOOD (Structure-of-Arrays):
   nodes_x = CuArray(...)  # Coalesced access
   nodes_y = CuArray(...)
   nodes_z = CuArray(...)
   ```

3. **Fused kernels**

   ```julia
   # Compute residual AND Jacobian-vector product in one kernel
   function fused_residual_and_jv!(r, Jv, u, v, ...)
       # Reuse shape function evaluations, Jacobians, etc.
   end
   ```

### 5.2 Preconditioner Implementation

**Chebyshev-Jacobi (Easiest, Do First):**

```julia
# Precompute diagonal (matrix-free!)
D = CuVector{Float64}(undef, n)
compute_diagonal_kernel!(D, mesh, material)

# Chebyshev iteration (pure GPU)
function chebyshev_smooth!(x, r, D, num_iter, λ_min, λ_max)
    for i in 1:num_iter
        θ = chebyshev_coefficient(i, λ_min, λ_max)
        x .+= θ .* (D .\ r)
        r = b - apply_operator(x)  # Matrix-free!
    end
end
```

**Element-Block Jacobi (Medium difficulty):**

```julia
# Batched small dense solves
function element_block_jacobi_kernel!(x, r, elements, ...)
    elem = threadIdx().x + ...
    
    # Assemble local 12×12 matrix (4 nodes × 3 DOFs)
    K_local = compute_element_stiffness(elem)
    r_local = extract_element_residual(r, elem)
    
    # Local solve (batched LAPACK on GPU)
    x_local = K_local \ r_local
    
    # Scatter
    scatter_to_global!(x, x_local, elem)
end
```

### 5.3 Communication (Multi-GPU)

**Strategy:** Domain decomposition + GPU-Direct

```julia
using MPI, CUDA.NCCL

# Halo exchange (overlapped with compute)
@async begin
    # Pack halo
    pack_halo_kernel!(send_buffer, u, halo_nodes)
    
    # Communicate (GPU-direct, no CPU copy!)
    MPI.Isend(send_buffer, neighbor_rank)
    MPI.Irecv(recv_buffer, neighbor_rank)
    
    # Unpack halo
    unpack_halo_kernel!(u, recv_buffer, ghost_nodes)
end

# Meanwhile, compute interior (no dependencies)
compute_interior_kernel!(r_interior, u_interior)
```

---

## 6. Practical Default Stack (RECOMMENDED)

### 6.1 Formulation

- Small/finite strain J2 plasticity with consistent tangents
- Mortar ALM for contact
- F-bar or u-p for near-incompressibility

### 6.2 Outer Loop

- **Inexact Newton** with backtracking line search
- **Eisenstat-Walker** forcing terms
- **Pseudo-transient continuation (PTC)** for difficult startup
- **Anderson (m=5-10)** on ALM updates (not on Newton!)

### 6.3 Linear Solver

- **FGMRES** (flexible, allows variable preconditioning)
- Matrix-free Jacobian application
- Relative tolerance: Eisenstat-Walker adaptive

### 6.4 Preconditioner ⭐ KEY COMPONENT

- **P2→P1 p-coarsening** (same mesh)
- **Chebyshev-Jacobi smoother** (3-5 iterations)
- **h-coarsening** (if mesh hierarchy available)
- **Coarse solve:** AMG with elasticity near-nullspace or BDDC
- **Optional:** Add vertex-star Schwarz for strong smoother

### 6.5 GPU Implementation

- Matrix-free residual & J·v
- Warp-per-element kernels
- Batched dense solves for patches
- Overlapped MPI+GPU communication

---

## 7. Implementation Roadmap (Updated)

### Phase 1: Core Newton-Krylov (Current)

- [x] Matrix-free residual assembly
- [x] JFNK Jacobian-vector product
- [x] GMRES solve
- [ ] **Add line search** ← NEXT!
- [ ] **Add Eisenstat-Walker** ← NEXT!

### Phase 2: Preconditioning (CRITICAL - NEW PRIORITY!)

- [ ] Diagonal extraction (matrix-free)
- [ ] Chebyshev-Jacobi smoother
- [ ] P2→P1 coarsening operators
- [ ] GMG V-cycle
- [ ] Benchmark: iterations vs. no preconditioner

### Phase 3: Contact (After Phase 2)

- [ ] Mortar contact detection
- [ ] ALM outer loop
- [ ] Anderson acceleration on ALM
- [ ] Semi-smooth Newton (optional)

### Phase 4: Material Models

- [ ] Perfect plasticity (current)
- [ ] Consistent tangent computation
- [ ] J2 with hardening
- [ ] Near-incompressibility (F-bar)

### Phase 5: Multi-GPU

- [ ] Domain decomposition
- [ ] Halo exchange with GPU-Direct
- [ ] BDDC coarse solver
- [ ] Weak scaling benchmarks

### Phase 6: Advanced (Future)

- [ ] Nonlinear Schwarz (ASPIN)
- [ ] u-p mixed formulation
- [ ] Arc-length continuation (Riks)
- [ ] Adaptive mesh refinement

---

## 8. What This Changes in Our Current Work

### Immediate Actions

1. **Update Newton solver** (demos/newton_krylov_anderson_cpu.jl):

   ```julia
   # Add backtracking line search
   # Add Eisenstat-Walker tolerance
   # Keep Anderson for future ALM, but don't use it on Newton yet
   ```

2. **Create preconditioner module**:

   ```julia
   # src/preconditioners/chebyshev_jacobi.jl
   # src/preconditioners/element_block_jacobi.jl
   # src/preconditioners/gmg.jl
   ```

3. **Add diagonal extraction**:

   ```julia
   # Matrix-free diagonal computation
   function compute_diagonal!(d, mesh, material)
       for i in 1:n
           e_i = unit_vector(i)
           d[i] = (apply_operator(e_i))[i]
       end
   end
   ```

### Long-Term Strategy Shift

**Before (our naive plan):**

```text
Newton → Anderson → GMRES (unpreconditioned) → Done
```

**After (expert-validated):**

```text
ALM (Anderson-accelerated) →
  Newton (line search + E-W) →
    FGMRES (GMG-preconditioned) →
      Chebyshev-Jacobi smoothing →
        Matrix-free operator
```

**Complexity increase:** Yes, but necessary for real-world problems!

---

## 9. Key Takeaways

### ✅ What We Got Right

- Matrix-free Newton-Krylov (core approach validated!)
- GPU-first design philosophy
- Plasticity with state variables
- Anderson acceleration (just wrong placement)

### 🔧 What We Must Add

1. **Globalization** (line search) - prevents divergence
2. **Preconditioning** (GMG) - THE critical performance factor
3. **Eisenstat-Walker** - reduces wasted work
4. **Correct Anderson placement** - on ALM/PTC, not Newton

### 🎯 Success Metrics (Revised)

- **Without preconditioner:** 50-100 GMRES iterations per Newton step (current)
- **With Jacobi:** 20-40 GMRES iterations (easy win)
- **With Chebyshev-Jacobi:** 10-20 GMRES iterations (big win)
- **With GMG:** 5-10 GMRES iterations (production-ready!)

---

## 10. References

**Expert feedback validates approaches from:**

- Knoll & Keyes (2004): "Jacobian-free Newton-Krylov methods"
- Eisenstat & Walker (1996): "Choosing the forcing terms"
- Briggs et al. (2000): "A Multigrid Tutorial"
- Toselli & Widlund (2005): "Domain Decomposition Methods"
- Anderson (1965): "Iterative procedures for nonlinear equations"

**Modern GPU implementations:**

- MFEM (mfem.org) - GMG on GPUs
- HYPRE (hypre.llnl.gov) - AMG with near-nullspace
- PETSc (petsc.org) - FGMRES + field-split preconditioners

---

**Bottom Line:** Our core direction is **correct**. We need to invest in **GPU-friendly preconditioning** (Chebyshev-Jacobi → GMG) and **globalization** (line search, E-W). Anderson stays but moves to ALM/PTC outer loops. This is the proven path to production-quality GPU nonlinear FEM.
