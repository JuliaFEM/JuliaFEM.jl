---
title: "GPU Assembly Architecture: Matrix-Free Kernel Design"
date: 2025-11-10
author: "Jukka Aho"
status: "Critical Design Decision"
last_updated: 2025-11-10
tags: ["gpu", "architecture", "matrix-free", "kernel-design"]
---

## The Fundamental Question

**User's insight:** "We need to go inside GPU right away, and exit it only to save some results or control some iterations."

This is the **correct** architectural principle. But how do we actually implement it?

## What Matrix-Free Newton-Krylov Actually Needs

Matrix-free only needs **one operation**: Compute `r = R(u)` and `Jv ≈ [R(u + εv) - R(u)] / ε`

```julia
# Traditional (BAD for GPU):
K = assemble_stiffness(elements, u)  # CPU assembly
f = assemble_forces(elements, u)     # CPU assembly
r = K*u - f                          # Transfer to GPU
Jv = K*v                             # Solve on GPU

# Matrix-Free (GOOD for GPU):
r = compute_residual_gpu!(u)         # EVERYTHING on GPU
Jv = compute_Jv_gpu!(u, v)          # EVERYTHING on GPU
```

## Two Parallelization Strategies

### Strategy A: Element-Level Parallelism (Traditional)

```julia
# One thread per element
@cuda threads=256 blocks=ceil(Int, n_elements/256) function residual_kernel!(
    r_global,              # Output: (n_dofs,) residual vector
    u_global,              # Input: (n_dofs,) DOF vector
    elements,              # Element connectivity
    node_coords,           # Node positions
    material_states        # Material state per integration point
)
    elem_id = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if elem_id > n_elements
        return
    end
    
    # Each thread processes ONE element
    elem = elements[elem_id]
    u_elem = get_element_dofs(u_global, elem)
    
    # Compute element residual
    r_elem = zeros(8)  # 8 DOFs for Quad4
    for ip in 1:4
        ξ, η, w = quadrature_points[ip]
        B = strain_displacement_matrix(elem, ξ, η)
        ε = B * u_elem
        
        # Material model evaluation
        state = material_states[elem_id, ip]
        σ, state_new = compute_stress(material, ε, state)
        
        r_elem += B' * σ * w * det_J
    end
    
    # PROBLEM: Atomic scatter (race condition!)
    for i in 1:8
        dof = elem.dofs[i]
        CUDA.@atomic r_global[dof] += r_elem[i]
    end
end
```

**Problems:**
1. ❌ **Atomic scatter** - Massive contention at shared DOFs
2. ❌ **Load imbalance** - Quad4 vs Hex27 = 4 vs 27 integration points
3. ❌ **Divergence** - Different elements have different shapes/materials
4. ⚠️ **Material state storage** - How to handle plasticity history?

### Strategy B: Node-Level Parallelism (Jukka's Research Idea!)

```julia
# One thread per DOF (node × direction)
@cuda threads=256 blocks=ceil(Int, n_dofs/256) function residual_kernel_nodal!(
    r_global,              # Output: (n_dofs,) residual vector
    u_global,              # Input: (n_dofs,) DOF vector
    node_to_elements,      # Which elements touch this node?
    elements,              # Element connectivity
    node_coords,           # Node positions
    material_states        # Material state per integration point
)
    dof_id = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if dof_id > n_dofs
        return
    end
    
    node_id = (dof_id - 1) ÷ 2 + 1  # 2D: 2 DOFs per node
    direction = (dof_id - 1) % 2 + 1
    
    # Accumulate residual from ALL elements touching this node
    r_accum = 0.0
    for elem_id in node_to_elements[node_id]
        elem = elements[elem_id]
        u_elem = get_element_dofs(u_global, elem)
        
        # Loop over integration points
        for ip in 1:n_ip
            ξ, η, w = quadrature_points[ip]
            B = strain_displacement_matrix(elem, ξ, η)
            ε = B * u_elem
            
            state = material_states[elem_id, ip]
            σ, state_new = compute_stress(material, ε, state)
            
            # Extract contribution to THIS DOF
            local_node_id = find_local_node(elem, node_id)
            r_accum += B[direction, local_node_id] * σ[direction] * w * det_J
        end
    end
    
    # NO ATOMICS! Each thread writes to unique location
    r_global[dof_id] = r_accum
end
```

**Advantages:**
1. ✅ **No atomic scatter** - Each DOF has one owner thread
2. ✅ **Coalesced writes** - Sequential DOF ordering
3. ✅ **Natural for contact** - Contact is node-based!
4. ✅ **Load balancing** - Nodes have similar valence

**Problems:**
1. ❌ **Redundant computation** - Same element computed by 4/8/20 threads
2. ❌ **Complex indexing** - Need `node_to_elements` map
3. ❌ **Material state updates** - Who updates plasticity state?

## The Material State Problem

**Critical issue:** Plasticity has history-dependent state at integration points.

```julia
struct PlasticState
    ε_p::SymmetricTensor{2,3}  # Plastic strain (6 components)
    α::Float64                  # Hardening parameter
    # ...
end
```

**In element-parallel:**
- Natural: One thread per element owns its integration point states
- Update: `material_states[elem_id, ip] = state_new`

**In node-parallel:**
- Problem: Multiple threads compute same element, who updates state?
- Solution 1: Atomic updates (slow, wrong - CAS not defined for structs)
- Solution 2: Separate state update pass (two kernel launches)
- Solution 3: Element-ownership even in node-parallel

## Hybrid Strategy: Node Residual + Element State

```julia
# Pass 1: Compute residual (node-parallel, read-only material states)
@cuda residual_kernel_nodal!(r, u, node_to_elements, elements, states)

# Pass 2: Update material states (element-parallel, write states)
@cuda state_update_kernel!(states_new, u, elements, states_old, converged_flags)
```

**Advantages:**
1. ✅ Residual computation is read-only → safe for node-parallel
2. ✅ State updates are element-parallel → natural ownership
3. ✅ Can skip state update during GMRES iterations (only after Newton step accepted)

**Cost:**
- Two kernel launches per residual evaluation
- Material model computed twice (once for residual, once for state)

## What About GMRES on GPU?

GMRES needs `Jv` product. Two options:

### Option 1: Finite Difference (Simple)
```julia
function compute_Jv_gpu!(Jv, u, v, ε=1e-8)
    # r0 = R(u) already computed
    u_perturbed = u .+ ε .* v  # GPU vector op
    r_perturbed = compute_residual_gpu!(u_perturbed)  # GPU kernel
    Jv .= (r_perturbed .- r0) ./ ε  # GPU vector op
end
```

### Option 2: Analytical (Complex)
```julia
function compute_Jv_gpu!(Jv, u, v, material_tangent)
    # Need to compute ∂R/∂u · v directly
    # Requires material tangent modulus at each IP
    # More accurate but requires AD or manual derivatives
end
```

**Recommendation:** Start with Option 1 (FD). If accuracy issues, switch to Option 2.

## Memory Layout for GPU

**Critical:** Structure-of-Arrays (SoA) for coalescing

```julia
# BAD (Array-of-Structs):
struct Element
    nodes::NTuple{4, Int}
    # ...
end
elements = Vector{Element}(...)  # NOT coalesced!

# GOOD (Structure-of-Arrays):
struct ElementData
    node1::Vector{Int}  # (n_elements,)
    node2::Vector{Int}  # (n_elements,)
    node3::Vector{Int}  # (n_elements,)
    node4::Vector{Int}  # (n_elements,)
    # ...
end
```

**Material states also need SoA:**

```julia
# BAD:
states = Matrix{PlasticState}(n_elements, n_ip)  # Struct not coalesced

# GOOD:
struct MaterialStates
    ε_p::Matrix{Float64}  # (n_elements × n_ip, 6) - strain components
    α::Vector{Float64}    # (n_elements × n_ip,)   - hardening
end
```

## Recommended Architecture

### Phase 1: Element-Parallel (Easier, Validate Correctness)

1. Implement element-parallel kernels with atomic scatter
2. Validate against CPU assembly
3. Accept atomic contention cost for now
4. Focus on getting material models working on GPU

### Phase 2: Optimize Atomics (Practical Improvement)

1. Use `CuSparseMatrixCSC` for scatter pattern
2. Launch one warp per element, reduce within warp, single atomic per DOF
3. Should reduce atomic contention by 32× (warp size)

### Phase 3: Node-Parallel (Research, If Needed)

1. Implement node-parallel residual kernel
2. Separate state update pass
3. Measure if redundant computation cost < atomic cost

## Implementation Roadmap

### Week 1: CPU AssemblyState (Current Plan - Keep!)
- `AssemblyState` struct with flat arrays
- Zero-allocation assembly on CPU
- Eisenstat-Walker Newton
- Baseline performance measurement

### Week 2-3: Element-Parallel GPU Kernel
```julia
function compute_residual_gpu!(
    r::CuVector{T},
    u::CuVector{T},
    elements::ElementData,    # SoA connectivity
    coords::CuMatrix{T},       # (n_nodes, dim)
    material_params::MaterialParams,  # E, ν, etc.
    material_states::MaterialStates   # ε_p, α (SoA)
)
    @cuda threads=256 blocks=n_blocks element_residual_kernel!(...)
end
```

### Week 4: GMRES Integration
```julia
using CUDA, Krylov

function solve_newton_gpu!(state::AssemblyState)
    u = CuArray(state.u)
    r = CUDA.zeros(n_dofs)
    
    for iter in 1:max_newton_iter
        # Compute residual on GPU
        compute_residual_gpu!(r, u, ...)
        
        # Matrix-free Jv operator
        Jv_op = MatrixFreeOperator(v -> compute_Jv_gpu!(u, v, ...))
        
        # GMRES on GPU
        du, stats = gmres(Jv_op, -r; atol=η_k)
        
        # Update on GPU
        u .+= du
    end
    
    # Copy result back
    state.u .= Array(u)
end
```

### Week 5+: Material Models on GPU

**Critical files to port:**
- `src/materials_plasticity.jl` - von Mises plasticity
- Need to rewrite return mapping algorithm for GPU

**Challenge:** Return mapping uses nested nonlinear solve!
```julia
function integrate_plasticity(ε, state_old)
    # Trial stress
    σ_trial = C : (ε - state_old.ε_p)
    f_trial = sqrt(3/2 * dev(σ_trial) : dev(σ_trial)) - σ_y
    
    if f_trial < 0
        return σ_trial, state_old  # Elastic
    else
        # NONLINEAR solve for Δλ (plastic multiplier)
        # This is expensive! Can't nest GMRES inside GMRES!
    end
end
```

**Solution:** Implement closed-form return mapping (if possible) or simple fixed-point iteration.

## Open Questions

1. **Material state storage:** How to handle 100s of bytes per IP efficiently on GPU?
2. **Contact:** How to integrate contact constraints into this framework?
3. **Adaptive quadrature:** How to handle variable n_ip per element?
4. **Mesh on GPU:** Should we keep mesh data on GPU permanently or transfer per solve?

## Key Insight from User

> "We cannot do cheap tricks like form the global stiffness matrix first outside and then just move everything to gpu and solve, it's not going to give us performance."

**This is correct.** The entire Newton loop must live on GPU:

```julia
# CPU controls loop, GPU does computation
u_gpu = CuArray(u0)
for iter in 1:max_iter
    r_gpu = compute_residual_gpu!(u_gpu)      # GPU kernel
    du_gpu = solve_matrix_free_gpu!(u_gpu)    # GPU GMRES
    u_gpu .+= du_gpu                          # GPU vector op
    
    if norm(r_gpu) < tol
        break
    end
end
u_final = Array(u_gpu)  # Only copy at end
```

Only CPU↔GPU transfers:
- Input: Initial guess `u0`
- Output: Final solution `u_final`
- Control: Convergence checks (can do on GPU with `CUDA.@allowscalar`)

## Next Steps

1. **User decision:** Element-parallel vs node-parallel vs hybrid?
2. **Material priority:** Which physics first? (Elasticity easiest, plasticity hardest)
3. **GMRES library:** Krylov.jl on GPU or custom implementation?

## Recommendation

Start with **element-parallel + atomic scatter** because:
1. Easier to implement and debug
2. Natural material state ownership
3. Atomic overhead may be acceptable with warp reduction
4. Can optimize later if bottleneck

Then measure. If atomics are bottleneck, consider node-parallel. If not, we're done!
