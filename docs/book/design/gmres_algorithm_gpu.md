---
title: "GMRES Algorithm for GPU: Matrix-Free Implementation"
date: 2025-11-11
author: "JuliaFEM Team"
status: "Authoritative"
last_updated: 2025-11-11
tags: ["gmres", "gpu", "matrix-free", "arnoldi", "krylov"]
---

## Why GMRES Instead of CG?

### The Problem with CG

**Conjugate Gradient (CG)** only works for **symmetric positive definite** systems:

```julia
# CG requires:
K = K'        # Symmetric
λ_min > 0     # Positive definite
```

### Real FEM Problems Are Unsymmetric

**Sources of unsymmetry:**

1. **Contact mechanics** - One-sided constraints (normal contact)
2. **Friction** - Tangential forces (Coulomb friction)
3. **Plasticity** - Tangent stiffness from return mapping
4. **Large deformation** - Geometric nonlinearity
5. **Stabilization** - SUPG, PSPG terms

**Example: Contact stiffness contribution**

```text
Master node i can push on slave node j
But slave node j CANNOT push back on master node i
→ K[i,j] ≠ K[j,i]  (UNSYMMETRIC!)
```

### GMRES: The Universal Solver

**GMRES (Generalized Minimal Residual)** works for **ANY invertible system**:

- Symmetric or unsymmetric ✅
- Positive definite or indefinite ✅
- Real or complex ✅

**Key advantage:** Start using it NOW, never need to switch later!

---

## GMRES Algorithm Overview

### Core Idea

**Minimize residual in Krylov subspace:**

```text
At iteration k, find x_k ∈ span{r₀, Ar₀, A²r₀, ..., Aᵏ⁻¹r₀} that minimizes ||b - Ax_k||
```

**How it works:**

1. **Build Krylov basis** {v₁, v₂, ..., vₖ} via Arnoldi iteration
2. **Form small Hessenberg matrix** H (k×k) representing A in Krylov space
3. **Solve least-squares problem** min ||β*e₁ - H*y||
4. **Reconstruct solution** x_k = x₀ + V*y

**Key insight:** All expensive work (Arnoldi orthogonalization) can be done on GPU with cuBLAS!

---

## Algorithm Breakdown

### 1. Arnoldi Iteration (Building Krylov Basis)

**Goal:** Construct orthonormal basis V = [v₁, v₂, ..., vₘ] for Krylov subspace

**Algorithm:**

```julia
v₁ = r₀ / ||r₀||  # Initial vector (normalized residual)

for j = 1:m
    # Matrix-vector product (YOUR tangent operator!)
    w = A * vⱼ
    
    # Modified Gram-Schmidt orthogonalization
    for i = 1:j
        hᵢⱼ = ⟨w, vᵢ⟩      # Inner product
        w = w - hᵢⱼ * vᵢ   # Subtract projection
    end
    
    hⱼ₊₁,ⱼ = ||w||
    vⱼ₊₁ = w / hⱼ₊₁,ⱼ
end
```

**Result:** 
- Orthonormal basis: V (n × m matrix)
- Hessenberg matrix: H (m+1 × m matrix, upper Hessenberg)

**GPU optimization:**
- All vectors (v₁, v₂, ..., vₘ) stay on GPU as CuArray
- Inner products: `CUBLAS.dot()`
- Vector updates: `CUBLAS.axpy!()` (w ← w - α*v)
- Norms: `CUBLAS.nrm2()`

### 2. Givens Rotations (QR Factorization)

**Goal:** Convert Hessenberg matrix H to upper triangular R via Givens rotations

**Why QR?** Transforms least-squares problem into triangular system (easy to solve!)

**Givens rotation:** Eliminates one subdiagonal element

```text
[  c   s ] [ h_i,j   ]   [ r_i,j ]
[ -s   c ] [ h_i+1,j ] = [   0   ]

where c² + s² = 1
```

**Formulas:**

```julia
function compute_givens(a, b)
    if b == 0
        return 1.0, 0.0
    end
    
    if abs(b) > abs(a)
        τ = -a / b
        s = 1 / sqrt(1 + τ²)
        c = s * τ
    else
        τ = -b / a
        c = 1 / sqrt(1 + τ²)
        s = c * τ
    end
    
    return c, s
end

function apply_givens!(H, c, s, i, j)
    temp = c * H[i, j] - s * H[i+1, j]
    H[i+1, j] = s * H[i, j] + c * H[i+1, j]
    H[i, j] = temp
end
```

**Apply incrementally:** After computing column j of H, apply all previous rotations, then compute new rotation to eliminate H[j+1, j].

**Result:** Upper triangular system R*y = β*e₁

### 3. Least-Squares Solve

**Problem:** min ||β*e₁ - H*y||

After Givens rotations: H = Q*R (QR factorization)

**Transformed problem:** R*y = Qᵀ(β*e₁) = s

where s is updated incrementally during Givens rotations.

**Solution:** Backward substitution (upper triangular system)

```julia
function solve_upper_triangular!(y, R, s, j)
    # Solve R[1:j, 1:j] * y[1:j] = s[1:j]
    for i = j:-1:1
        y[i] = s[i]
        for k = (i+1):j
            y[i] -= R[i, k] * y[k]
        end
        y[i] /= R[i, i]
    end
end
```

**GPU:** Small matrix (m × m where m ≈ 30), can use cuSOLVER or just do on CPU (data transfer negligible)

### 4. Solution Reconstruction

**Recover solution:** x_k = x₀ + V*y

```julia
# V is n × k matrix (Krylov basis)
# y is k-vector (least-squares solution)
x = x₀ + V[:, 1:k] * y
```

**GPU:** Use `CUBLAS.gemv!('N', 1.0, V, y, 1.0, x)` for matrix-vector product

---

## Matrix-Free GMRES for Newton-Krylov

### The Two Different Operations

**CRITICAL DISTINCTION:** There are TWO operations in Newton-Krylov:

#### Operation 1: Residual Evaluation (NOT a matvec!)

```julia
# Compute out-of-balance forces
f_int = assemble_internal_forces(u)  # ∫ Bᵀ σ(u) dV
r = f_ext - f_int                    # Residual vector

# For linear elasticity:
r = f_ext - K*u  # But this is computed via element assembly, not matvec!
```

**This is:** Element-by-element assembly to get internal forces

**NOT:** A matrix-vector product (no K*v operation here!)

#### Operation 2: Tangent Matvec (what GMRES needs!)

```julia
# Compute tangent stiffness times direction vector
w = K_t(u) * v  # This IS a matrix-vector product!

# For linear elasticity:
w = K * v  # Tangent is constant

# For nonlinear:
w = K_t(u) * v  # Tangent depends on current state
```

**This is:** Matrix-vector product K_t * v (computed matrix-free)

**GMRES calls this:** 30-50 times per Newton iteration!

### Newton's Equation Breakdown

**Newton step:** `J(u) * Δu = -r(u)`

**For elasticity:**
- Jacobian: `J = ∂r/∂u = ∂(f_ext - f_int)/∂u = -∂f_int/∂u = K_t`
- Equation becomes: `K_t(u) * Δu = -(f_ext - f_int(u))`

**Or rearranged:** `K_t(u) * Δu = f_int(u) - f_ext`

For **linear elasticity** where `f_int(u) = K*u`:
```julia
K * Δu = K*u - f_ext
K * (u + Δu) = K*u + K*Δu = f_ext  # Standard equilibrium!
```

### The Full Picture

**Newton iteration:** Solve J(u)*Δu = -r(u)

**Matrix-free approach:** Never form Jacobian J explicitly!

**Instead:** Provide matrix-vector product operator `w = J*v`

```julia
# Option 1: Finite difference approximation (general, slower)
function jacobian_matvec(u, v)
    ε = sqrt(eps()) * norm(u) / norm(v)
    r_perturbed = residual(u + ε*v)
    r_current = residual(u)
    return (r_perturbed - r_current) / ε
end

# Option 2: Direct tangent computation (faster, what we use!)
function jacobian_matvec(u, v)
    # For elasticity: J = K_t, so J*v = K_t*v
    return tangent_stiffness_matvec(u, v)
end
```

**For linear elasticity:**

```julia
# Tangent operator: K*v (element-by-element assembly)
function tangent_matvec(u, v)
    y = zeros(n_dofs)
    for element in elements
        # Local tangent: K_local (8×8 for Tet4)
        K_local = compute_element_tangent(element, E, ν)
        
        # Extract local v
        v_local = v[element.dofs]
        
        # Local matvec
        y_local = K_local * v_local
        
        # Add to global
        y[element.dofs] += y_local
    end
    return y
end
```

**GPU version:** Same as current implementation, just pass to GMRES instead of CG!

### Concrete Example: What Each Operation Does

**Problem:** Cantilever beam with tip load

**Given:**
- Current displacement: `u = [0, 0, 0, ...]` (initial guess or Newton iterate)
- External forces: `f_ext = [0, -10, 0, ...]` (tip traction)
- Direction vector: `v = [1, 0, 0, 0.5, ...]` (from GMRES Arnoldi)

**Step 1: Compute residual (element assembly, NOT a matvec conceptually)**
```julia
f_int = zeros(n_dofs)
for element in elements
    u_local = u[element.dofs]           # 8 values (Tet4: 4 nodes × 3 DOFs)
    
    # Compute element internal forces
    σ = compute_stress(element, u_local, E, ν)
    f_int_local = ∫ Bᵀ σ dV            # 8 values
    
    f_int[element.dofs] += f_int_local
end

r = f_ext - f_int  # Residual (out-of-balance forces)
# For linear: r = f_ext - K*u (but computed via assembly, not explicit K)
```

**Step 2: Define tangent matvec (what GMRES repeatedly calls)**
```julia
function tangent_matvec(v)
    # v is a DIRECTION vector from GMRES Arnoldi iteration
    # We compute w = K * v (THIS IS THE MATVEC GMRES NEEDS!)
    
    w = zeros(n_dofs)
    for element in elements
        v_local = v[element.dofs]              # Extract local part: 8 values
        
        # Element tangent stiffness (constant for linear elasticity!)
        K_local = ∫ Bᵀ D B dV                  # 8×8 matrix
        
        w_local = K_local * v_local            # THIS IS THE MATVEC: 8 values
        
        w[element.dofs] += w_local             # Assemble to global
    end
    
    return w  # Result: w = K*v
end

# Example call:
w = tangent_matvec(v)  # Returns K*v
# If v = [1, 0, 0, 0.5, ...], then w = K*[1, 0, 0, 0.5, ...]
```

**Step 3: GMRES solves K*Δu = -r**
```julia
# GMRES builds Krylov subspace {v₁, K*v₁, K²*v₁, ...}
# by repeatedly calling: w = tangent_matvec(v)
# 
# Arnoldi iteration:
#   v₁ = r₀ / ||r₀||
#   w = tangent_matvec(v₁)  ← Called here! Computes K*v₁
#   Orthogonalize w against v₁
#   v₂ = w / ||w||
#   w = tangent_matvec(v₂)  ← Called again! Computes K*v₂
#   ...
#
# Total calls: 30-50 per Newton iteration

Δu = gmres(tangent_matvec, -r, tol=1e-6)
```

**Step 4: Update**
```julia
u_new = u + Δu  # New displacement
```

### Summary Table

| Operation | Formula | When Called | Input | Output | Purpose |
|-----------|---------|-------------|-------|--------|---------|
| **Residual** | `r = f_ext - f_int(u)` | Once/Newton | `u` (current state) | `r` (out-of-balance) | Check convergence |
| **Tangent matvec** | `w = K_t(u) * v` | 30-50×/Newton | `v` (direction) | `w = K*v` | GMRES Arnoldi |

**Key insight:** 
- Residual uses current displacement `u` → gives residual vector `r`
- Matvec uses direction vector `v` → gives `K*v`
- **Different inputs, different purposes!**

**For linear elasticity:**
- Both involve same element loop
- Residual: `r = f_ext - K*u` (but `K*u` computed via stress integration)
- Matvec: `w = K*v` (computed via stiffness matrix times vector)
- Same `K`, but operating on different vectors!

---

## Complete GPU-Resident GMRES

### Implementation Structure

```julia
"""
    gmres_gpu!(x, matvec_op, b; m=30, tol=1e-6, max_iter=100)

GPU-resident GMRES solver.

# Arguments
- `x::CuVector`: Initial guess (modified in-place)
- `matvec_op(v)`: Function computing A*v (returns CuVector)
- `b::CuVector`: Right-hand side
- `m::Int`: Restart parameter (Krylov subspace dimension)
- `tol::Float64`: Convergence tolerance
- `max_iter::Int`: Maximum iterations

# Returns
- `(iterations, residual_norm, converged)`
"""
function gmres_gpu!(
    x::CuVector{Float64},
    matvec_op::Function,
    b::CuVector{Float64};
    m::Int = 30,
    tol::Float64 = 1e-6,
    max_iter::Int = 100
)
    n = length(b)
    
    # Allocate Krylov workspace on GPU
    V = CUDA.zeros(Float64, n, m+1)    # Orthonormal basis
    H = CUDA.zeros(Float64, m+1, m)    # Upper Hessenberg
    cs = CUDA.zeros(Float64, m)        # Givens cosines
    sn = CUDA.zeros(Float64, m)        # Givens sines
    s = CUDA.zeros(Float64, m+1)       # RHS for least squares
    y = CUDA.zeros(Float64, m)         # Least squares solution
    
    # Temporary vectors
    r = CUDA.similar(b)
    w = CUDA.similar(b)
    
    iter = 0
    
    while iter < max_iter
        # Compute initial residual: r = b - A*x
        r .= matvec_op(x)
        r .= b .- r
        β = CUDA.norm(r)
        
        # Check convergence
        if β < tol
            return (iter, β, true)
        end
        
        # Arnoldi iteration
        V[:, 1] .= r ./ β
        s[1] = β
        s[2:end] .= 0.0
        
        for j in 1:m
            iter += 1
            
            # Matrix-vector product
            w .= matvec_op(view(V, :, j))
            
            # Modified Gram-Schmidt orthogonalization
            for i in 1:j
                H[i, j] = CUDA.dot(w, view(V, :, i))
                CUDA.axpy!(-H[i, j], view(V, :, i), w)
            end
            
            H[j+1, j] = CUDA.norm(w)
            
            if H[j+1, j] > 1e-14
                V[:, j+1] .= w ./ H[j+1, j]
            end
            
            # Apply previous Givens rotations
            for i in 1:(j-1)
                apply_givens!(H, cs[i], sn[i], i, j)
            end
            
            # Compute new Givens rotation
            cs[j], sn[j] = compute_givens(H[j, j], H[j+1, j])
            
            # Apply to H and s
            apply_givens!(H, cs[j], sn[j], j, j)
            apply_givens_to_rhs!(s, cs[j], sn[j], j)
            
            # Check residual
            β = abs(s[j+1])
            
            if β < tol || iter >= max_iter
                # Solve least squares
                solve_upper_triangular!(y, H, s, j)
                
                # Update solution: x += V[:, 1:j] * y
                CUDA.gemv!('N', 1.0, view(V, :, 1:j), view(y, 1:j), 1.0, x)
                
                return (iter, β, β < tol)
            end
        end
        
        # GMRES(m) restart
        solve_upper_triangular!(y, H, s, m)
        CUDA.gemv!('N', 1.0, view(V, :, 1:m), view(y, 1:m), 1.0, x)
    end
    
    return (max_iter, norm(b - matvec_op(x)), false)
end

# Helper functions (small, can be on GPU or CPU)
function apply_givens!(H, c, s, i, j)
    temp = c * H[i, j] - s * H[i+1, j]
    H[i+1, j] = s * H[i, j] + c * H[i+1, j]
    H[i, j] = temp
end

function apply_givens_to_rhs!(s, c, s_coeff, i)
    temp = c * s[i] - s_coeff * s[i+1]
    s[i+1] = s_coeff * s[i] + c * s[i+1]
    s[i] = temp
end

function compute_givens(a, b)
    if abs(b) < 1e-14
        return 1.0, 0.0
    end
    
    if abs(b) > abs(a)
        τ = -a / b
        s = 1 / sqrt(1 + τ^2)
        c = s * τ
    else
        τ = -b / a
        c = 1 / sqrt(1 + τ^2)
        s = c * τ
    end
    
    return c, s
end

function solve_upper_triangular!(y, R, s, k)
    for i in k:-1:1
        y[i] = s[i]
        for j in (i+1):k
            y[i] -= R[i, j] * y[j]
        end
        y[i] /= R[i, i]
    end
end
```

---

## Integration with Newton-Krylov

### Inexact Newton-Krylov with GMRES

**Replace CG with GMRES everywhere:**

```julia
function solve_newton_gmres_gpu!(
    gpu_data::ElasticityDataGPU,
    physics::Physics;
    newton_tol = 1e-6,
    max_newton = 20,
    gmres_restart = 30,
    max_gmres_per_newton = 50,
    forcing_power = 0.5,
    forcing_max = 0.9
)
    u = gpu_data.u
    n_dofs = length(u)
    
    total_gmres_iters = 0
    history = Tuple{Int,Float64,Float64}[]
    
    for newton_iter in 1:max_newton
        # 1. Compute residual
        f_int = compute_residual_gpu!(gpu_data, u)
        R = gpu_data.f_ext - f_int
        apply_dirichlet_to_vector!(R, gpu_data.is_fixed)
        
        R_norm = norm(R)
        
        # Check convergence
        if R_norm < newton_tol
            return (u, newton_iter, total_gmres_iters, R_norm, history)
        end
        
        # 2. Adaptive forcing (Eisenstat-Walker)
        η = min(forcing_max, R_norm^forcing_power)
        gmres_tol = η * R_norm
        
        # 3. Solve K*Δu = -R using GMRES (matrix-free!)
        Δu = CUDA.zeros(Float64, n_dofs)
        
        function tangent_matvec(v)
            return tangent_operator_gpu(gpu_data, u, v)
        end
        
        gmres_iters, gmres_residual, converged = gmres_gpu!(
            Δu,
            tangent_matvec,
            -R;
            m = gmres_restart,
            tol = gmres_tol,
            max_iter = max_gmres_per_newton
        )
        
        total_gmres_iters += gmres_iters
        push!(history, (gmres_iters, R_norm, η))
        
        # 4. Update solution
        u .+= Δu
        apply_dirichlet_to_vector!(u, gpu_data.is_fixed)
    end
    
    return (u, max_newton, total_gmres_iters, history[end][2], history)
end
```

---

## Performance Characteristics

### CG vs GMRES Comparison

**For SYMMETRIC systems:**

| Method | Storage | Work per iter | Typical iters |
|--------|---------|---------------|---------------|
| CG | 4 vectors | 1 matvec + 2 dots | 50-200 |
| GMRES(30) | 32 vectors | 1 matvec + 30 dots | 30-100 |

**Verdict:** CG slightly cheaper per iteration, but similar overall cost

**For UNSYMMETRIC systems:**

| Method | Works? | Storage | Work per iter |
|--------|--------|---------|---------------|
| CG | ❌ FAILS | - | - |
| GMRES(30) | ✅ WORKS | 32 vectors | 1 matvec + 30 dots |

**Verdict:** GMRES is ONLY option!

### GPU Memory Requirements

**GMRES(m=30) workspace:**

- Krylov basis V: n × 31 vectors (largest allocation)
- Hessenberg H: 31 × 30 = 930 floats (negligible)
- Other: ~5 vectors (r, w, s, cs, sn)

**Total: ~36 × n_dofs × 8 bytes**

**Example:** 1M DOFs → 288 MB (fits easily on modern GPUs)

### Restart Parameter Tuning

**m = restart parameter (Krylov subspace dimension)**

**Tradeoffs:**

- **Small m (10-20):** Less memory, more restarts, slower convergence
- **Large m (50-100):** More memory, fewer restarts, faster convergence
- **Sweet spot: m = 30** (good balance)

**For contact/friction:** May need larger m (50-80) due to ill-conditioning

---

## Advantages for Contact Mechanics

### 1. Handles Unsymmetry Naturally

**Contact stiffness is inherently unsymmetric:**

```text
Master surface pushes on slave → K[slave, master] ≠ 0
Slave cannot push on master → K[master, slave] = 0
```

**GMRES:** Doesn't care about symmetry!

### 2. Matrix-Free = Easy Active Set Changes

**Newton iteration:**

```julia
for newton_iter in 1:max_newton
    # Update active set (which contacts are active)
    update_contact_status!(gpu_data, u)
    
    # Tangent includes current active set
    function tangent_with_contact(v)
        K_v = elastic_tangent_matvec(v)
        C_v = contact_tangent_matvec(v)  # Only active contacts!
        return K_v + C_v
    end
    
    # GMRES just calls tangent_with_contact
    gmres_gpu!(Δu, tangent_with_contact, -R)
end
```

**No matrix reassembly!** Active set changes = different matvec results

### 3. Preconditioning

**GMRES works with right preconditioning:**

```julia
# Solve (A*M⁻¹)*(M*x) = b
# Preconditioner M approximates A⁻¹

gmres_gpu!(z, v -> matvec(precondition(v)), b)
x = precondition(z)
```

**For contact:** Diagonal Jacobi or block-Jacobi (node-level blocks)

---

## Summary

### Key Takeaways

✅ **GMRES handles unsymmetric systems** (CG fails)
✅ **Contact/friction are unsymmetric** (need GMRES)
✅ **Matrix-free via matvec operator** (no assembly)
✅ **GPU-friendly**: cuBLAS for orthogonalization
✅ **Restart parameter m=30** balances memory/speed
✅ **Integrates with Newton-Krylov** (Eisenstat-Walker forcing)

### Implementation Roadmap

**Phase 1: Replace CG with GMRES** (current)
- Drop-in replacement: `gmres_gpu!()` instead of `cg_solve_matfree_gpu!()`
- Same tangent operator
- Test on linear elasticity (should match CG results)

**Phase 2: Add Contact** (next)
- Implement contact detection on GPU
- Add contact tangent to matvec operator
- Test on Hertz contact problem

**Phase 3: Add Friction** (later)
- Coulomb friction model
- Augmented Lagrangian or penalty
- Unsymmetric tangent (GMRES shines here!)

**Phase 4: Preconditioning** (optimization)
- Diagonal Jacobi (easiest)
- Block-Jacobi (better convergence)
- ILU(0) (best, but harder on GPU)

---

## References

1. Saad & Schultz (1986): "GMRES: A generalized minimal residual algorithm"
2. Kelley (1995): "Iterative Methods for Linear and Nonlinear Equations"
3. Your own docs: `docs/src/book/multigpu_nodal_assembly.md`
4. Your own blog: `docs/src/book/blog/krylov_nodal_assembly.jl`

**Bottom line:** Use GMRES from day 1. It's the right tool for contact mechanics!
