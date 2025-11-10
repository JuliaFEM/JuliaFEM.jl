---
title: "GPU Implementation Strategy: Practical Steps"
date: 2025-11-10
author: "Jukka Aho + AI"
status: "Implementation Plan"
last_updated: 2025-11-10
tags: ["gpu", "implementation", "roadmap", "kernels"]
---

## Core Principle

**Everything stays on GPU except initial input and final output.**

```julia
# WRONG: Ping-pong between CPU and GPU
for elem in elements
    u_elem = u_global[elem.nodes]        # CPU
    r_elem = compute_element(u_elem)     # CPU
    r_global[elem.nodes] += r_elem       # CPU
end
u_gpu = CuArray(u_global)                # Transfer
solve!(u_gpu)                            # GPU

# RIGHT: Stay on GPU entire Newton loop
u_gpu = CuArray(u0)                      # Initial transfer
for newton_iter in 1:max_iter
    r_gpu = compute_residual_gpu!(u_gpu)     # GPU kernel
    du_gpu = gmres_gpu!(r_gpu, u_gpu)        # GPU solver
    u_gpu .+= du_gpu                         # GPU op
end
u_final = Array(u_gpu)                   # Final transfer
```

## The Minimal Viable GPU Kernel

For elasticity, we need kernel that computes residual: `r = ∫ Bᵀ σ dV`

```julia
@cuda threads=256 blocks=n_blocks function elasticity_residual_kernel!(
    r_global,              # (n_dofs,) - OUTPUT
    u_global,              # (n_dofs,) - INPUT
    elem_nodes,            # (n_elements, nodes_per_elem) - CONNECTIVITY
    node_coords,           # (n_nodes, dim) - GEOMETRY
    E, ν                   # Material parameters
)
    elem_id = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if elem_id > n_elements
        return
    end
    
    # Local element residual (8 DOFs for Quad4)
    r_local = MVector{8, Float64}(zeros(8))
    
    # Get element nodes
    nodes = (elem_nodes[elem_id, 1], 
             elem_nodes[elem_id, 2],
             elem_nodes[elem_id, 3], 
             elem_nodes[elem_id, 4])
    
    # Get element DOFs from global vector
    u_local = get_element_dofs(u_global, nodes)
    
    # Quadrature loop
    for ip in 1:4
        ξ, η = gauss_points[ip]
        w = gauss_weights[ip]
        
        # Shape function derivatives
        dN_dξ = shape_derivatives(ξ, η)
        
        # Jacobian
        J = compute_jacobian(dN_dξ, node_coords, nodes)
        dN_dx = J \ dN_dξ
        det_J = det(J)
        
        # B-matrix (strain-displacement)
        B = assemble_B_matrix(dN_dx)
        
        # Strain
        ε = B * u_local
        
        # Stress (linear elasticity)
        C = constitutive_matrix(E, ν)
        σ = C * ε
        
        # Accumulate to local residual
        r_local .+= B' * σ * w * det_J
    end
    
    # Scatter to global (ATOMIC for shared DOFs)
    for i in 1:8
        dof = get_global_dof(nodes, i)
        CUDA.@atomic r_global[dof] += r_local[i]
    end
end
```

## Data Layout: Everything as Flat Arrays

**Key insight:** GPU kernels need contiguous memory, not Julia structs with pointers.

```julia
struct GPUAssemblyData{T}
    # Mesh topology (SoA for coalescing)
    elem_nodes::CuMatrix{Int32}        # (n_elements, 4) for Quad4
    node_coords::CuMatrix{T}           # (n_nodes, 2) for 2D
    
    # Material parameters (constant on GPU)
    E::T
    ν::T
    
    # DOF vector (lives on GPU entire solve)
    u::CuVector{T}                     # (n_dofs,)
    r::CuVector{T}                     # (n_dofs,)
    
    # Plasticity state (if needed)
    ε_p::CuMatrix{T}                   # (n_elements * n_ip, 6)
    α::CuVector{T}                     # (n_elements * n_ip,)
end
```

## Material State: The Hard Problem

Plasticity needs history at each integration point:

```julia
# Option 1: Flatten everything (SoA)
struct PlasticStateGPU{T}
    ε_p_xx::CuVector{T}  # (n_elements * n_ip,)
    ε_p_yy::CuVector{T}
    ε_p_zz::CuVector{T}
    ε_p_xy::CuVector{T}
    ε_p_yz::CuVector{T}
    ε_p_xz::CuVector{T}
    α::CuVector{T}       # Hardening parameter
end

# Access in kernel:
function get_plastic_strain(states, elem_id, ip)
    idx = (elem_id - 1) * 4 + ip  # 4 IPs for Quad4
    return (states.ε_p_xx[idx],
            states.ε_p_yy[idx],
            states.ε_p_zz[idx],
            states.ε_p_xy[idx],
            states.ε_p_yz[idx],
            states.ε_p_xz[idx])
end

# Option 2: Matrix storage (easier, less optimal)
ε_p::CuMatrix{T}  # (n_elements * n_ip, 6)

# Access in kernel:
function get_plastic_strain(ε_p_matrix, elem_id, ip)
    idx = (elem_id - 1) * 4 + ip
    return (ε_p_matrix[idx, 1],
            ε_p_matrix[idx, 2],
            ε_p_matrix[idx, 3],
            ε_p_matrix[idx, 4],
            ε_p_matrix[idx, 5],
            ε_p_matrix[idx, 6])
end
```

**Recommendation:** Start with Option 2 (matrix), optimize to Option 1 if needed.

## Handling Atomics: Warp-Level Reduction

Instead of 8 atomic adds per element, do warp reduction first:

```julia
@cuda threads=256 blocks=n_blocks function residual_kernel_optimized!(
    r_global, u_global, ...
)
    # One warp (32 threads) per element
    warp_id = (threadIdx().x - 1) ÷ 32 + 1
    lane_id = (threadIdx().x - 1) % 32 + 1
    elem_id = warp_id + (blockIdx().x - 1) * (blockDim().x ÷ 32)
    
    if elem_id > n_elements
        return
    end
    
    # Each thread computes subset of local residual
    r_local_thread = zeros(8)
    if lane_id <= 4  # 4 integration points
        ip = lane_id
        # ... compute contribution from this IP ...
        r_local_thread .= B' * σ * w * det_J
    end
    
    # Warp reduction (add across threads)
    r_local_reduced = warp_reduce(r_local_thread)
    
    # Only one thread does atomic scatter
    if lane_id == 1
        for i in 1:8
            dof = get_global_dof(nodes, i)
            CUDA.@atomic r_global[dof] += r_local_reduced[i]
        end
    end
end
```

**Benefit:** 32× fewer atomic operations!

## Matrix-Free Jacobian-Vector Product

For GMRES, we need `Jv ≈ [R(u + εv) - R(u)] / ε`:

```julia
function compute_Jv_gpu!(
    Jv::CuVector{T},
    u::CuVector{T},
    v::CuVector{T},
    asm_data::GPUAssemblyData{T},
    ε::T = 1e-7
)
    # r0 already computed
    r0 = asm_data.r
    
    # Perturb u
    u_perturbed = u .+ ε .* v  # GPU vector operation
    
    # Compute residual at perturbed state
    r_perturbed = CUDA.zeros(T, length(u))
    @cuda threads=256 blocks=n_blocks elasticity_residual_kernel!(
        r_perturbed, u_perturbed, ...
    )
    
    # Finite difference approximation
    Jv .= (r_perturbed .- r0) ./ ε
end
```

**Cost:** 2× residual evaluations per GMRES iteration, but no matrix assembly!

## GMRES on GPU: Use Krylov.jl

```julia
using Krylov, CUDA

function solve_newton_gpu!(asm_data::GPUAssemblyData)
    u = asm_data.u
    r = asm_data.r
    
    for iter in 1:max_newton_iter
        # Compute residual on GPU
        @cuda threads=256 blocks=n_blocks elasticity_residual_kernel!(
            r, u, asm_data.elem_nodes, asm_data.node_coords, 
            asm_data.E, asm_data.ν
        )
        
        # Check convergence
        r_norm = CUDA.norm(r)
        if r_norm < tol
            break
        end
        
        # Matrix-free operator for GMRES
        function matvec!(Jv, v)
            compute_Jv_gpu!(Jv, u, CuVector(v), asm_data)
        end
        A_op = LinearOperator(Float64, length(u), length(u), false, false,
                              (y, v) -> matvec!(y, v))
        
        # GMRES solve on GPU
        du, stats = gmres(A_op, -r, atol=1e-6, rtol=1e-6)
        
        # Update solution
        u .+= CuVector(du)
    end
end
```

## Plasticity Return Mapping on GPU

**Challenge:** Return mapping is iterative nonlinear solve!

```julia
@cuda function plasticity_kernel!(...)
    # ... compute trial stress ...
    
    σ_trial = C : (ε - ε_p_old)
    f_trial = sqrt(3/2 * dev(σ_trial) : dev(σ_trial)) - σ_y
    
    if f_trial < 0
        # Elastic step
        σ = σ_trial
        ε_p_new = ε_p_old
    else
        # Plastic step - need to solve for Δλ
        # CANNOT use nested GMRES on GPU!
        
        # Option 1: Closed-form (if available)
        Δλ = f_trial / (3 * G + H)  # For linear hardening
        
        # Option 2: Fixed-point iteration (simple)
        Δλ = 0.0
        for sub_iter in 1:10
            # ... Newton iteration for Δλ ...
            # Must converge in few iterations!
        end
        
        # Update state
        N = dev(σ_trial) / norm(dev(σ_trial))
        ε_p_new = ε_p_old + Δλ * N
        σ = σ_trial - 2 * G * Δλ * N
    end
end
```

**Critical:** Return mapping must be cheap (no nested iterative solves).

## Practical Implementation Steps

### Step 1: CPU Baseline (Week 1)

Create CPU version with same flat data structure:

```julia
struct AssemblyData{T}
    elem_nodes::Matrix{Int32}
    node_coords::Matrix{T}
    E::T
    ν::T
    u::Vector{T}
    r::Vector{T}
end

function compute_residual_cpu!(data::AssemblyData)
    fill!(data.r, 0.0)
    for elem_id in 1:n_elements
        # Same logic as GPU kernel but on CPU
        # ...
    end
end
```

### Step 2: Port to GPU (Week 2)

```julia
# Convert to GPU arrays
gpu_data = GPUAssemblyData(
    CuArray(data.elem_nodes),
    CuArray(data.node_coords),
    data.E, data.ν,
    CuArray(data.u),
    CuArray(data.r)
)

# Launch kernel
@cuda threads=256 blocks=n_blocks elasticity_residual_kernel!(
    gpu_data.r, gpu_data.u, gpu_data.elem_nodes,
    gpu_data.node_coords, gpu_data.E, gpu_data.ν
)
```

### Step 3: Validate (Week 2)

```julia
# Compare CPU vs GPU
compute_residual_cpu!(cpu_data)
@cuda ... elasticity_residual_kernel!(gpu_data...)
r_gpu_cpu = Array(gpu_data.r)

@test r_gpu_cpu ≈ cpu_data.r rtol=1e-10
```

### Step 4: Newton Loop (Week 3)

```julia
function solve_nonlinear_gpu!(gpu_data)
    for iter in 1:20
        compute_residual_gpu!(gpu_data)
        r_norm = CUDA.norm(gpu_data.r)
        if r_norm < 1e-8
            break
        end
        
        # Solve linear system
        du = gmres_gpu!(gpu_data)
        gpu_data.u .+= du
    end
end
```

### Step 5: Matrix-Free (Week 4)

Replace direct solve with matrix-free GMRES using Krylov.jl

### Step 6: Plasticity (Week 5+)

Add material state arrays and plasticity kernel

## Performance Expectations

Based on benchmarks:

**Small problems (1K DOFs):** GPU slower due to overhead

**Medium problems (5K DOFs):** 3-4× GPU speedup

**Large problems (10K+ DOFs):** 5-10× GPU speedup

**Matrix-free benefit:** 3-8× reduction in Newton iteration cost

**Total speedup potential:** 15-80× for large problems

## Open Questions for User

1. **Start with element-parallel?** (Easier but has atomic contention)
2. **Which physics first?** (Linear elasticity easiest, plasticity hardest)
3. **Material state storage?** (Matrix storage or separate vectors?)
4. **Node-parallel eventually?** (Your research idea - needs more design)
5. **Contact mechanics?** (How to integrate penalty/Lagrange multipliers?)

## Key Takeaway

The architecture must be:

- **Kernel-based:** All physics inside GPU kernels
- **Flat arrays:** No nested structs with pointers
- **Resident data:** u, r, states stay on GPU entire solve
- **Matrix-free:** No global matrix assembly
- **Minimal atomics:** Use warp reduction when possible

This is fundamentally different from traditional FEM, but it's the only way to get real GPU performance.
