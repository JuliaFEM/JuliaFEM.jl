---
title: "GPU Kernel Design: Node-Parallel vs Element-Parallel"
date: 2025-11-10
author: "Jukka Aho"
status: "Design Exploration"
last_updated: 2025-11-10
tags: ["gpu", "kernels", "parallelization", "node-assembly"]
---

## User's Question

> "I can only think that we should have a kernel function which takes as input arguments some node i and then all the other necessary details like node_to_elements map, elements, material state, new material state, and things like that so that it can - do some magic."

This is asking about **node-level parallelism** - your research idea!

## Two Approaches Compared

### Approach A: Element-Parallel (Traditional GPU FEM)

```julia
# One thread per element
@cuda threads=256 blocks=ceil(Int, n_elements/256) function element_kernel!(
    r_global::CuDeviceVector{T},
    u_global::CuDeviceVector{T},
    elem_nodes::CuDeviceMatrix{Int32},  # (n_elements, 4)
    node_coords::CuDeviceMatrix{T},     # (n_nodes, 2)
    E::T, ν::T
)
    elem_id = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if elem_id > size(elem_nodes, 1)
        return
    end
    
    # Get element connectivity
    n1, n2, n3, n4 = elem_nodes[elem_id, 1], elem_nodes[elem_id, 2],
                     elem_nodes[elem_id, 3], elem_nodes[elem_id, 4]
    
    # Get element DOFs (8 DOFs for 2D Quad4)
    u_elem = (u_global[2*n1-1], u_global[2*n1],
              u_global[2*n2-1], u_global[2*n2],
              u_global[2*n3-1], u_global[2*n3],
              u_global[2*n4-1], u_global[2*n4])
    
    # Compute element residual (loop over 4 integration points)
    r_elem = MVector{8, T}(zeros(8))
    for ip in 1:4
        ξ, η = gauss_points_2x2[ip]
        w = gauss_weights_2x2[ip]
        
        # Shape function derivatives at (ξ, η)
        dN = shape_derivatives_quad4(ξ, η)
        
        # Jacobian matrix
        J = compute_jacobian_quad4(dN, node_coords, n1, n2, n3, n4)
        det_J = det_2x2(J)
        inv_J = inv_2x2(J)
        
        # Physical derivatives: dN/dx = inv(J) * dN/dξ
        dN_dx = inv_J * dN
        
        # B-matrix (strain-displacement matrix)
        B = assemble_B_matrix_2d(dN_dx)
        
        # Strain: ε = B * u_elem
        ε = B * u_elem  # (3×8) * (8×1) = (3×1): [εxx, εyy, γxy]
        
        # Constitutive matrix (plane strain)
        C = constitutive_matrix_2d(E, ν)
        
        # Stress: σ = C * ε
        σ = C * ε  # (3×3) * (3×1) = (3×1): [σxx, σyy, σxy]
        
        # Accumulate: r_elem += Bᵀ * σ * w * det(J)
        r_elem .+= B' * σ * (w * det_J)
    end
    
    # Scatter to global residual (ATOMIC - multiple elements share nodes!)
    CUDA.@atomic r_global[2*n1-1] += r_elem[1]
    CUDA.@atomic r_global[2*n1]   += r_elem[2]
    CUDA.@atomic r_global[2*n2-1] += r_elem[3]
    CUDA.@atomic r_global[2*n2]   += r_elem[4]
    CUDA.@atomic r_global[2*n3-1] += r_elem[5]
    CUDA.@atomic r_global[2*n3]   += r_elem[6]
    CUDA.@atomic r_global[2*n4-1] += r_elem[7]
    CUDA.@atomic r_global[2*n4]   += r_elem[8]
end
```

**Pros:**

- Natural material state ownership (one thread owns integration point states)
- Simple data structure (just element connectivity)
- Each element computed exactly once (no redundancy)

**Cons:**

- **8 atomic operations per element** (massive contention at shared DOFs!)
- Load imbalance if mixed element types (Tri3 vs Hex27)
- Divergence if different materials

### Approach B: Node-Parallel (Your Idea!)

```julia
# One thread per DOF (node × direction)
@cuda threads=256 blocks=ceil(Int, n_dofs/256) function node_kernel!(
    r_global::CuDeviceVector{T},
    u_global::CuDeviceVector{T},
    node_to_elems::CuDeviceVector{Int32},   # Flat array: [elem1, elem2, ...]
    node_to_elems_offsets::CuDeviceVector{Int32},  # CSR-like: node i has elements [offsets[i]:offsets[i+1]-1]
    elem_nodes::CuDeviceMatrix{Int32},
    node_coords::CuDeviceMatrix{T},
    E::T, ν::T
)
    dof_id = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if dof_id > length(r_global)
        return
    end
    
    # Which node and direction?
    node_id = (dof_id - 1) ÷ 2 + 1  # Integer division
    direction = (dof_id - 1) % 2 + 1  # 1 = x, 2 = y
    
    # Accumulate residual from all elements touching this node
    r_accum = 0.0
    
    # Get elements touching this node (CSR-like access)
    start_idx = node_to_elems_offsets[node_id]
    end_idx = node_to_elems_offsets[node_id + 1] - 1
    
    for elem_idx in start_idx:end_idx
        elem_id = node_to_elems[elem_idx]
        
        # Get element connectivity
        n1, n2, n3, n4 = elem_nodes[elem_id, 1], elem_nodes[elem_id, 2],
                         elem_nodes[elem_id, 3], elem_nodes[elem_id, 4]
        
        # Which local node are we? (1, 2, 3, or 4)
        local_node_id = if node_id == n1
            1
        elseif node_id == n2
            2
        elseif node_id == n3
            3
        else
            4
        end
        
        # Get element DOFs
        u_elem = (u_global[2*n1-1], u_global[2*n1],
                  u_global[2*n2-1], u_global[2*n2],
                  u_global[2*n3-1], u_global[2*n3],
                  u_global[2*n4-1], u_global[2*n4])
        
        # Loop over integration points
        for ip in 1:4
            ξ, η = gauss_points_2x2[ip]
            w = gauss_weights_2x2[ip]
            
            # Compute B-matrix (same as element-parallel)
            dN = shape_derivatives_quad4(ξ, η)
            J = compute_jacobian_quad4(dN, node_coords, n1, n2, n3, n4)
            det_J = det_2x2(J)
            inv_J = inv_2x2(J)
            dN_dx = inv_J * dN
            B = assemble_B_matrix_2d(dN_dx)
            
            # Strain and stress
            ε = B * u_elem
            C = constitutive_matrix_2d(E, ν)
            σ = C * ε
            
            # Extract contribution to THIS DOF
            # B is (3×8), we want column for this node and direction
            col_idx = 2 * (local_node_id - 1) + direction
            
            # r = Bᵀ * σ, so we want (Bᵀ)[:,col_idx] ⋅ σ = B[col_idx, :] ⋅ σ
            # Actually: r[col_idx] = sum(B[strain_comp, col_idx] * σ[strain_comp])
            r_contribution = 0.0
            for strain_comp in 1:3
                r_contribution += B[strain_comp, col_idx] * σ[strain_comp]
            end
            r_contribution *= w * det_J
            
            r_accum += r_contribution
        end
    end
    
    # Write to global (NO ATOMICS! Each DOF owned by one thread)
    r_global[dof_id] = r_accum
end
```

**Pros:**

- **Zero atomic operations!** Each DOF has unique owner thread
- Coalesced writes to `r_global` (sequential DOF ordering)
- Natural for contact mechanics (contact is node-based)
- Better load balancing (most nodes have similar valence)

**Cons:**

- **Redundant computation:** Same element computed by 4 threads (Quad4) or 8 (Hex8)
- Complex data structure: Need `node_to_elems` map (CSR format)
- Material state updates unclear (who updates plasticity state?)

## Data Structure for Node-Parallel: CSR Format

The key is `node_to_elems` map in CSR (Compressed Sparse Row) format:

```julia
# Example mesh: 4 nodes, 2 Quad4 elements
#
# Element 1: nodes [1, 2, 4, 3]
# Element 2: nodes [2, 5, 6, 4]
#
# Node connectivity:
# Node 1: [elem 1]
# Node 2: [elem 1, elem 2]
# Node 3: [elem 1]
# Node 4: [elem 1, elem 2]
# Node 5: [elem 2]
# Node 6: [elem 2]

# CSR representation:
node_to_elems = [1,  1, 2,  1,  1, 2,  2,  2]  # Flat list of elements
#                 ↑   ↑      ↑   ↑      ↑   ↑
#               node1 node2   node3 node4 node5 node6

node_to_elems_offsets = [1, 2, 4, 5, 7, 8, 9]
#                        ↑  ↑  ↑  ↑  ↑  ↑  ↑
#                       n1 n2 n3 n4 n5 n6 end

# Access elements for node i:
start = node_to_elems_offsets[i]
stop = node_to_elems_offsets[i+1] - 1
elems_touching_node_i = node_to_elems[start:stop]
```

**Building this on CPU:**

```julia
function build_node_to_elems_map(elem_nodes::Matrix{Int}, n_nodes::Int)
    # Count elements per node
    elem_count = zeros(Int, n_nodes)
    for elem in 1:size(elem_nodes, 1)
        for local_node in 1:size(elem_nodes, 2)
            node = elem_nodes[elem, local_node]
            elem_count[node] += 1
        end
    end
    
    # Build offsets (cumulative sum)
    offsets = zeros(Int, n_nodes + 1)
    offsets[1] = 1
    for i in 1:n_nodes
        offsets[i+1] = offsets[i] + elem_count[i]
    end
    
    # Fill element list
    node_to_elems = zeros(Int, offsets[end] - 1)
    current_pos = copy(offsets[1:end-1])
    for elem in 1:size(elem_nodes, 1)
        for local_node in 1:size(elem_nodes, 2)
            node = elem_nodes[elem, local_node]
            pos = current_pos[node]
            node_to_elems[pos] = elem
            current_pos[node] += 1
        end
    end
    
    return node_to_elems, offsets
end
```

**Transfer to GPU:**

```julia
node_to_elems_gpu = CuArray(node_to_elems)
offsets_gpu = CuArray(offsets)
```

## Material State Problem: Hybrid Approach

**Problem:** In node-parallel, 4 threads compute the same element. Who updates plasticity state?

**Solution:** Separate state update pass (element-parallel)

```julia
# Pass 1: Compute residual (node-parallel, read-only states)
@cuda node_kernel!(r, u, node_to_elems, offsets, elem_nodes, coords, 
                   material_states_old, E, ν)

# Pass 2: Update material states (element-parallel, write states)
@cuda element_state_update_kernel!(material_states_new, u, elem_nodes,
                                   coords, material_states_old, E, ν)
```

**Key insight:** State update only needed when Newton step is ACCEPTED, not during GMRES iterations!

```julia
for newton_iter in 1:max_iter
    # Compute residual (uses old states)
    @cuda node_kernel!(r, u, ..., material_states_old, ...)
    
    # GMRES iterations (many calls to node_kernel, same states)
    for gmres_iter in 1:max_gmres_iter
        compute_Jv!(Jv, u, v, material_states_old)  # Still uses old states
    end
    
    # Accept Newton step
    u .+= du
    
    # NOW update material states (only once per Newton iteration)
    @cuda element_state_update_kernel!(material_states_new, u, ...)
    material_states_old .= material_states_new
end
```

## Performance Trade-off

**Element-Parallel:**

- Computation: 1× (each element computed once)
- Atomics: 8 per element = high contention
- Memory: Simple structure

**Node-Parallel:**

- Computation: 4× (each element computed by 4 threads for Quad4)
- Atomics: 0 (no contention!)
- Memory: Extra CSR structure (~20 bytes per node)

**Which is faster?** Depends on:

1. Atomic contention cost (depends on mesh topology)
2. Arithmetic intensity (cheap ops → atomics dominate, expensive ops → computation dominates)
3. Element type (Tri3: 3× redundancy, Hex27: 27× redundancy!)

## Recommendation for JuliaFEM

**Phase 1: Element-Parallel (Now)**

- Easier to implement and debug
- Material state ownership is natural
- Can optimize atomics with warp reduction
- Validates GPU assembly correctness

**Phase 2: Node-Parallel (Later, if needed)**

- Your research idea!
- Implement after element-parallel working
- Compare performance on realistic meshes
- May excel for contact problems (contact is node-based)

## Hybrid Strategy: Best of Both?

```julia
# Small elements (Tri3, Tet4): Element-parallel
# - Low atomic contention (3-4 DOFs)
# - Redundancy cost too high

# Large elements (Hex27): Node-parallel
# - 27× redundancy in node-parallel unacceptable
# - But 20 atomic writes per element also bad!

# Solution: Dispatch based on element type
if element_type == Tri3 || element_type == Tet4
    @cuda element_kernel!(...)
else
    @cuda node_kernel!(...)
end
```

## My Strong Recommendation

**Start with element-parallel + warp reduction:**

```julia
# One warp (32 threads) per element
# Threads cooperate to compute element, then ONE atomic per DOF
@cuda threads=256 blocks=n_blocks function warp_element_kernel!(...)
    warp_id = (threadIdx().x - 1) ÷ 32 + 1
    lane_id = (threadIdx().x - 1) % 32 + 1
    elem_id = warp_id + ...
    
    # Distribute work among warp
    if lane_id <= n_integration_points
        ip = lane_id
        r_ip = compute_contribution_at_ip(ip, ...)
    end
    
    # Warp reduction (sum across threads)
    r_elem = warp_reduce_sum(r_ip)
    
    # Only lane 1 does atomic scatter
    if lane_id == 1
        for i in 1:8
            CUDA.@atomic r_global[...] += r_elem[i]
        end
    end
end
```

**This gives:**

- 1× computation (no redundancy)
- 1× atomics per DOF (32× reduction!)
- Simple data structure
- Natural state ownership

Then measure. If still bottleneck, try node-parallel.

## Your GMRES Question

> "It's actually a bit unclear me that how we're actually going to implement this krylov gmres thing."

**Answer:** Use Krylov.jl with matrix-free operator:

```julia
using Krylov, CUDA

# Matrix-free operator: computes Jv = [R(u+εv) - R(u)] / ε
struct GPUMatrixFreeOp{T}
    u::CuVector{T}
    r0::CuVector{T}
    # ... mesh data ...
end

function LinearAlgebra.mul!(Jv, op::GPUMatrixFreeOp, v)
    ε = 1e-7
    u_pert = op.u .+ ε .* CuVector(v)
    
    # Compute residual on GPU
    r_pert = CUDA.zeros(length(u_pert))
    @cuda node_kernel!(r_pert, u_pert, ...)  # Or element_kernel!
    
    Jv .= (r_pert .- op.r0) ./ ε
end

# Solve
op = GPUMatrixFreeOp(u, r, ...)
du, stats = gmres(op, -r, atol=1e-6)
```

**Key:** GMRES stays on GPU, only needs Jv product!

## Summary

**Your instinct is correct:** We need kernels that compute physics on GPU, not matrix assembly on CPU.

**Two paths forward:**

1. **Element-parallel** (easier, natural state ownership, optimize atomics)
2. **Node-parallel** (your research idea, no atomics, needs careful design)

**My vote:** Element-parallel with warp reduction first. Prove it works. Then experiment with node-parallel.

What do you think? Should we proceed with element-parallel kernel implementation next week?
