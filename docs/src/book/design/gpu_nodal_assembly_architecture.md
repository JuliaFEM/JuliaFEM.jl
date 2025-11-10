---
title: "GPU Nodal Assembly Architecture"
date: 2025-11-10
author: "Jukka Aho"
status: "Authoritative"
tags: ["gpu", "nodal-assembly", "architecture"]
---

## Core Principles

### 1. Tensors.jl Throughout

**✅ CuArray{SymmetricTensor{2,3}, 1} Works!**

```julia
# Store tensors directly on GPU
σ_gp = CuArray{SymmetricTensor{2,3,Float64,6}}(undef, n_gp)
ε_p_gp = CuArray{SymmetricTensor{2,3,Float64,6}}(undef, n_gp)
F_gp = CuArray{Tensor{2,3,Float64,9}}(undef, n_gp)

# Natural operations in kernels
ε = symmetric(sum(dN_dx[i] ⊗ u[i] for i in 1:4))
σ = λ * tr(ε) * I + 2μ * ε
f_node = dN ⋅ σ
```

**No Voigt notation needed! Natural tensor indexing everywhere!**

### 2. Nodal Assembly (Matrix-Free)

**❌ WRONG (element-based, needs atomics):**
```julia
for elem in elements
    compute element forces
    CUDA.@atomic r[node] += f_elem[i]  # Contention!
end
```

**✅ RIGHT (node-based, no atomics):**
```julia
for node in nodes
    f_node = sum over elements touching node
    r[node] = f_node  # Direct write, no atomics!
end
```

**Benefits:**
- No atomic operations (each thread owns a node)
- Matrix-free (consume GP data immediately)
- Contact-friendly (contact IS nodal)
- Cache-efficient (process node data together)

---

## Two-Phase Pipeline

### Phase 1: Integration Point Data (Material State Update)

**One thread per integration point - perfectly parallel!**

```julia
@cuda threads=256 blocks=ceil(Int, n_gp/256) compute_gp_data_kernel!(
    σ_gp,           # CuArray{SymmetricTensor{2,3,Float64,6}, 1}
    states_new,     # CuArray{PlasticState, 1}
    u,              # CuArray{Float64, 1}
    nodes,          # CuArray{Float64, 2} - shape (3, n_nodes)
    elements,       # CuArray{Int32, 2} - shape (4, n_elems)
    states_old,     # CuArray{PlasticState, 1}
    E, ν, σ_y       # Material parameters
)
```

**Kernel logic (per GP):**

```julia
function compute_gp_data_kernel!(σ_gp, states_new, u, nodes, elements, states_old, E, ν, σ_y)
    gp_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if gp_idx <= length(σ_gp)
        # Map GP to element and local GP
        elem_idx = (gp_idx - 1) ÷ 4 + 1  # 4 GPs per Tet4
        local_gp = (gp_idx - 1) % 4 + 1
        
        # Extract element nodes (using Tensors.jl!)
        n1, n2, n3, n4 = elements[1, elem_idx], elements[2, elem_idx], 
                         elements[3, elem_idx], elements[4, elem_idx]
        
        X1 = Vec{3}((nodes[1, n1], nodes[2, n1], nodes[3, n1]))
        X2 = Vec{3}((nodes[1, n2], nodes[2, n2], nodes[3, n2]))
        X3 = Vec{3}((nodes[1, n3], nodes[2, n3], nodes[3, n3]))
        X4 = Vec{3}((nodes[1, n4], nodes[2, n4], nodes[3, n4]))
        
        u1 = Vec{3}((u[3*n1-2], u[3*n1-1], u[3*n1]))
        u2 = Vec{3}((u[3*n2-2], u[3*n2-1], u[3*n2]))
        u3 = Vec{3}((u[3*n3-2], u[3*n3-1], u[3*n3]))
        u4 = Vec{3}((u[3*n4-2], u[3*n4-1], u[3*n4]))
        
        # Shape derivatives (constant for Tet4)
        dN1 = Vec{3}((-1.0, -1.0, -1.0))
        dN2 = Vec{3}((1.0, 0.0, 0.0))
        dN3 = Vec{3}((0.0, 1.0, 0.0))
        dN4 = Vec{3}((0.0, 0.0, 1.0))
        
        # Jacobian (using tensor products!)
        J = dN1 ⊗ X1 + dN2 ⊗ X2 + dN3 ⊗ X3 + dN4 ⊗ X4
        invJ = inv(J)
        
        # Physical derivatives (using tensor contractions!)
        dN1_dx = invJ ⋅ dN1
        dN2_dx = invJ ⋅ dN2
        dN3_dx = invJ ⋅ dN3
        dN4_dx = invJ ⋅ dN4
        
        # Strain (using tensor products and symmetric!)
        ε = symmetric(dN1_dx ⊗ u1 + dN2_dx ⊗ u2 + dN3_dx ⊗ u3 + dN4_dx ⊗ u4)
        
        # Material state update (using Tensors.jl!)
        state_old = states_old[gp_idx]
        σ, state_new = return_mapping_tensor(ε, state_old, E, ν, σ_y)
        
        # Store results
        σ_gp[gp_idx] = σ
        states_new[gp_idx] = state_new
    end
    
    return nothing
end
```

**Key features:**
- Uses Tensors.jl throughout (⊗, ⋅, symmetric, inv)
- Each GP independent - perfect parallelism
- Expensive plasticity computation isolated here
- No assembly - just compute and store

### Phase 2: Nodal Assembly (Matrix-Free)

**One thread per node - no atomics needed!**

```julia
@cuda threads=256 blocks=ceil(Int, n_nodes/256) nodal_assembly_kernel!(
    r,              # CuArray{Float64, 1} - residual vector
    σ_gp,           # CuArray{SymmetricTensor{2,3,Float64,6}, 1}
    u,              # CuArray{Float64, 1}
    nodes,          # CuArray{Float64, 2}
    elements,       # CuArray{Int32, 2}
    node_to_elems   # NodeToElementsMap (CSR format)
)
```

**Kernel logic (per node):**

```julia
function nodal_assembly_kernel!(r, σ_gp, u, nodes, elements, node_to_elems)
    node_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if node_idx <= size(nodes, 2)
        # Accumulate forces from all elements touching this node
        f_node = zero(Vec{3,Float64})
        
        # Get element range for this node (CSR format)
        elem_start = node_to_elems.ptr[node_idx]
        elem_end = node_to_elems.ptr[node_idx + 1] - 1
        
        # Loop over touching elements
        for elem_offset in elem_start:elem_end
            elem_idx = node_to_elems.data[elem_offset]
            
            # Find local node index in element
            local_node = find_local_node_index(node_idx, elem_idx, elements)
            
            # Loop over GPs in this element
            for local_gp in 1:4  # 4 GPs for Tet4
                gp_idx = (elem_idx - 1) * 4 + local_gp
                
                # Get stress at this GP
                σ = σ_gp[gp_idx]
                
                # Recompute dN/dx for this node at this GP
                # (Could precompute and store, but matrix-free approach recomputes)
                dN_dx = compute_dN_dx_for_node(local_node, local_gp, elem_idx, nodes, elements)
                
                # Gauss weight and detJ
                w = gauss_weight(local_gp)
                detJ = compute_detJ(elem_idx, nodes, elements)
                
                # Accumulate force contribution (using tensor contraction!)
                f_node += (dN_dx ⋅ σ) * (w * detJ)
            end
        end
        
        # Write result (no atomics - this node is ours!)
        r[3*node_idx-2] = f_node[1]
        r[3*node_idx-1] = f_node[2]
        r[3*node_idx]   = f_node[3]
    end
    
    return nothing
end
```

**Key features:**
- Each thread owns a node (no atomics!)
- Matrix-free: recompute geometry, consume stress immediately
- Uses Tensors.jl for contraction: `dN ⋅ σ`
- Natural for contact (which is nodal)

---

## Data Structures

### NodeToElementsMap (CSR Format)

**Stores which elements touch each node:**

```julia
struct NodeToElementsMap
    ptr::CuArray{Int32, 1}   # Length: n_nodes + 1
    data::CuArray{Int32, 1}  # Length: total node-element connections
end

# Example: Node 5 touches elements [12, 17, 23, 31]
# ptr[5] = 10      (start index in data)
# ptr[6] = 14      (end index - 1)
# data[10:13] = [12, 17, 23, 31]
```

**Build once on CPU, transfer to GPU:**

```julia
function build_node_to_elems(elements::Matrix{Int}, n_nodes::Int)
    # Count connections per node
    counts = zeros(Int, n_nodes)
    for elem in eachcol(elements)
        for node in elem
            counts[node] += 1
        end
    end
    
    # Build CSR
    ptr = cumsum([1; counts])
    data = Vector{Int32}(undef, sum(counts))
    
    # Fill data
    offset = copy(ptr[1:end-1])
    for (elem_idx, elem) in enumerate(eachcol(elements))
        for node in elem
            data[offset[node]] = elem_idx
            offset[node] += 1
        end
    end
    
    return NodeToElementsMap(CuArray(ptr), CuArray(data))
end
```

### PlasticState (GPU-compatible)

```julia
struct PlasticState
    ε_p::SymmetricTensor{2,3,Float64,6}  # Plastic strain (Tensors.jl!)
    α::Float64                            # Accumulated plastic strain
end

# Can store directly in CuArray!
states = CuArray{PlasticState}(undef, n_gp)
```

---

## Material Model (Using Tensors.jl)

### Return Mapping for Perfect Plasticity

```julia
function return_mapping_tensor(ε_total::SymmetricTensor{2,3,T},
                               state_old::PlasticState,
                               E, ν, σ_y) where T
    # Elastic strain
    ε_e = ε_total - state_old.ε_p
    
    # Elastic predictor (using Tensors.jl!)
    λ = E * ν / ((1 + ν) * (1 - 2ν))
    μ = E / (2(1 + ν))
    I = one(ε_e)
    σ_trial = λ * tr(ε_e) * I + 2μ * ε_e
    
    # Deviatoric stress (using Tensors.jl!)
    σ_dev = dev(σ_trial)
    σ_eq = sqrt(3/2 * (σ_dev ⊡ σ_dev))
    
    # Yield check
    f = σ_eq - σ_y
    
    if f <= 0.0
        # Elastic
        return (σ_trial, state_old)
    else
        # Plastic - radial return
        Δγ = f / (3μ)
        n = σ_dev / σ_eq
        
        σ = σ_trial - 2μ * Δγ * n
        
        # Update state
        Δε_p = Δγ * n
        ε_p_new = state_old.ε_p + Δε_p
        α_new = state_old.α + Δγ
        
        state_new = PlasticState(ε_p_new, α_new)
        
        return (σ, state_new)
    end
end
```

**Everything uses Tensors.jl - natural and efficient!**

---

## Complete Pipeline

```julia
function residual_gpu!(r, u, mesh, material, states_old, node_to_elems)
    n_gp = length(states_old)
    n_nodes = size(mesh.nodes, 2)
    
    # Phase 1: Compute all integration point data
    σ_gp = CuArray{SymmetricTensor{2,3,Float64,6}}(undef, n_gp)
    states_new = CuArray{PlasticState}(undef, n_gp)
    
    threads = 256
    blocks = ceil(Int, n_gp / threads)
    
    @cuda threads=threads blocks=blocks compute_gp_data_kernel!(
        σ_gp, states_new,
        u, mesh.nodes_gpu, mesh.elements_gpu,
        states_old,
        material.E, material.ν, material.σ_y
    )
    
    # Phase 2: Nodal assembly (matrix-free!)
    fill!(r, 0.0)
    
    threads = 256
    blocks = ceil(Int, n_nodes / threads)
    
    @cuda threads=threads blocks=blocks nodal_assembly_kernel!(
        r, σ_gp,
        u, mesh.nodes_gpu, mesh.elements_gpu,
        node_to_elems
    )
    
    return r, states_new
end
```

---

## Performance Characteristics

### Phase 1: Integration Point Kernel

**Workload per thread:**
- Extract 4 nodes × 3 coords (coalesced reads)
- Extract 4 displacements × 3 DOFs (coalesced reads)
- Compute Jacobian (tensor products): ~50 FLOPs
- Compute strain: ~50 FLOPs
- Return mapping: ~100-500 FLOPs (material dependent)
- Write 1 stress tensor + 1 state: coalesced writes

**Bottleneck:** Return mapping computation (CPU-bound, not memory-bound)

**Parallelism:** Perfect - all GPs independent

### Phase 2: Nodal Assembly Kernel

**Workload per thread:**
- Read node_to_elems (CSR, somewhat irregular)
- For each touching element:
  - Recompute geometry (matrix-free!)
  - Read stress (coalesced if GPs ordered by element)
  - Compute force contribution: ~30 FLOPs
- Write 3 DOFs: coalesced

**Bottleneck:** Irregular memory access (node_to_elems traversal)

**Parallelism:** Perfect - all nodes independent, no atomics!

---

## Advantages of This Design

### 1. No Atomic Operations
- Each thread owns exactly one node
- Direct writes, no contention
- Better performance, simpler code

### 2. Matrix-Free
- Don't store element stiffness matrices
- Recompute geometry on the fly
- Lower memory footprint
- Natural for nonlinear problems

### 3. Tensors.jl Throughout
- Natural tensor operations (⊗, ⋅, symmetric, dev, etc.)
- No manual Voigt notation
- Easier to read and maintain
- Type-stable and efficient

### 4. Contact-Ready
- Contact forces are nodal
- Natural integration with nodal assembly
- No special handling needed

### 5. Scalable
- Phase 1: scales with number of integration points
- Phase 2: scales with number of nodes
- Both perfectly parallel

---

## Implementation Phases

### Phase 1: CPU Reference
- Implement nodal assembly on CPU
- Use Tensors.jl throughout
- Validate correctness

### Phase 2: GPU Port - Phase 1 Kernel
- Port GP data computation to GPU
- Test against CPU
- Benchmark performance

### Phase 3: GPU Port - Phase 2 Kernel
- Port nodal assembly to GPU
- Test against CPU
- Benchmark performance

### Phase 4: Integration
- Combine both kernels
- End-to-end testing
- Performance optimization

### Phase 5: Newton-Krylov Integration
- Matrix-free Jacobian-vector products
- GMRES with GPU arrays
- Complete nonlinear solver

---

## Memory Layout Best Practices

### Nodes (Structure-of-Arrays)
```julia
# Store as 3 × n_nodes for coalesced access
nodes = CuArray{Float64, 2}(undef, 3, n_nodes)
# All X-coordinates contiguous, all Y-coordinates contiguous, etc.
```

### Elements (Column-Major)
```julia
# Store as 4 × n_elems for coalesced access
elements = CuArray{Int32, 2}(undef, 4, n_elems)
# Element 1: elements[:, 1] = [n1, n2, n3, n4]
```

### Solution Vector (Interleaved)
```julia
# DOFs interleaved: [ux1, uy1, uz1, ux2, uy2, uz2, ...]
u = CuArray{Float64, 1}(undef, 3 * n_nodes)
```

### Integration Point Data (Array of Tensors)
```julia
# Tensors.jl types work in CuArray!
σ_gp = CuArray{SymmetricTensor{2,3,Float64,6}, 1}(undef, n_gp)
states = CuArray{PlasticState, 1}(undef, n_gp)
```

---

**This is the architecture. Clean, efficient, and ready to implement!**
