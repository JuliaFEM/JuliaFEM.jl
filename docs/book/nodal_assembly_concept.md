---
title: "Nodal Assembly: Concept and Data Structures"
date: 2025-11-11
author: "JuliaFEM Team"
status: "Experimental"
last_updated: 2025-11-11
tags: ["assembly", "nodal", "gpu", "architecture"]
---

## Introduction

This document describes the **nodal assembly** concept - an alternative to traditional element-based assembly that is naturally suited for:

- GPU parallelization (no atomic operations needed)
- Matrix-free methods (Krylov solvers)
- Contact mechanics (contact is inherently nodal)
- Domain decomposition (nodes have clear ownership)

**Status:** Experimental concept with working prototype. See `src/nodal_assembly_structures.jl` and tests.

## The Problem with Element Assembly

Traditional FEM assembles **element by element**:

```julia
# Traditional element assembly
for element in elements
    K_local = compute_element_stiffness(element)  # 30×30 for Tet10
    
    # Scatter to global (requires atomic operations on GPU!)
    for i in 1:ndofs_local, j in 1:ndofs_local
        K_global[gdof[i], gdof[j]] += K_local[i,j]  # Race condition!
    end
end
```

**Problems:**

1. **GPU:** Multiple elements write to same global DOF → need atomics → slow
2. **Contact:** Contact forces are nodal, but assembly is elemental → mismatch
3. **Matrix-free:** Hard to compute K*v without forming K

## Nodal Assembly Solution

Assemble **node by node** instead:

```julia
# Nodal assembly
for node_i in nodes
    # Compute contributions FROM all elements touching node_i
    K_blocks, f_int = compute_nodal_contribution(node_i, elements_touching_i)
    
    # Each thread owns its node → no atomics needed!
    w[3*(node_i-1)+1:3*node_i] = matvec_nodal(K_blocks, u)
end
```

**Advantages:**

1. **GPU:** One thread per node, no conflicts, no atomics
2. **Contact:** Natural fit (contact forces already nodal)
3. **Matrix-free:** Direct K*v computation without forming global K

## The "Spider" Pattern

For node $i$, we only compute stiffness blocks for nodes it couples with:

```text
        j₃
        /\
       /  \
      /    \
    j₂------i------j₄    ← Node i's "spider"
      \    /
       \  /
        \/
        j₁
```

**Key insight:** Most nodes couple with only ~10-30 neighbors (not all N nodes!)

- **Corner node:** 8 neighbors (1 element touches it)
- **Interior node:** 27 neighbors (8 elements touch it)  
- **Face node:** 12 neighbors (intermediate)

**Efficiency:** Sparse connectivity preserved without storing full matrix!

## Data Structures

### 1. Inverse Mapping: Node → Elements

```julia
struct ElementNodeInfo
    element_id::Int           # Which element
    local_node_idx::Int       # Which local node index (1-10 for Tet10)
end

struct NodeToElementsMap
    node_to_elements::Vector{Vector{ElementNodeInfo}}
    nnodes::Int
    nelements::Int
end

# Usage
map = NodeToElementsMap(connectivity)
for elem_info in map.node_to_elements[node_i]
    println("Node $node_i is local node $(elem_info.local_node_idx) ",
            "in element $(elem_info.element_id)")
end
```

**Purpose:** Given node, find all elements touching it (needed for nodal loop).

### 2. Spider Nodes

```julia
function get_node_spider(map::NodeToElementsMap, node_id::Int, 
                         connectivity) -> Vector{Int}
    spider = Set{Int}()
    
    # Union of all nodes in elements touching node_id
    for elem_info in map.node_to_elements[node_id]
        for node in connectivity[elem_info.element_id]
            push!(spider, node)
        end
    end
    
    return sort(collect(spider))
end
```

**Purpose:** Find all nodes that couple with `node_id` (non-zero stiffness blocks).

### 3. Nodal Stiffness Contribution

```julia
struct NodalStiffnessContribution{T}
    node_id::Int
    spider_nodes::Vector{Int}              # Nodes that couple
    K_blocks::Vector{Tensor{2,3,T}}        # 3×3 blocks (one per spider node)
    f_int::Vec{3,T}                        # Internal force at this node
    f_ext::Vec{3,T}                        # External force at this node
end
```

**Purpose:** Storage for nodal assembly. `K_blocks[k]` is the 3×3 coupling between `node_id` and `spider_nodes[k]`.

**Zero-allocation:** All quantities use `Tensors.jl` types (immutable, stack-allocated).

## Matrix-Free Matvec

Given nodal contributions, compute $\mathbf{w} = \mathbf{K} \mathbf{u}$ without forming $\mathbf{K}$:

```julia
function matrix_vector_product_nodal(contrib::NodalStiffnessContribution, 
                                     u::Vector{Vec{3}}) -> Vec{3}
    w = zero(Vec{3})
    
    # Loop over spider nodes (only non-zero columns!)
    for (k, node_j) in enumerate(contrib.spider_nodes)
        K_ij = contrib.K_blocks[k]  # 3×3 block
        u_j = u[node_j]              # Displacement at node j
        
        w += K_ij ⋅ u_j  # Block matvec
    end
    
    return w
end
```

**Performance:**

- Only computes non-zero contributions (sparse spider)
- Zero allocations (Tensors.jl)
- GPU-friendly (parallel over nodes)

## Example: 2 Tet4 Elements

```text
Mesh:
  Element 1: nodes (1,2,3,4)
  Element 2: nodes (2,3,4,5)
  
  Nodes 2,3,4 shared between elements
```

**Node 1 (corner):**

- Touches: 1 element
- Spider: [1, 2, 3, 4]  (4 nodes)
- Needs: 4 × 3×3 blocks

**Node 2 (interior):**

- Touches: 2 elements
- Spider: [1, 2, 3, 4, 5]  (5 nodes = union of both elements)
- Needs: 5 × 3×3 blocks

**Node 5 (corner):**

- Touches: 1 element
- Spider: [2, 3, 4, 5]  (4 nodes)
- Needs: 4 × 3×3 blocks

## Assembly Algorithm

```julia
# 1. Build inverse mapping (once, at mesh creation)
map = NodeToElementsMap(connectivity)

# 2. For each node (parallel on GPU)
for node_i in 1:nnodes
    # Find spider
    spider = get_node_spider(map, node_i, connectivity)
    
    # Allocate storage
    contrib = NodalStiffnessContribution(node_i, spider)
    
    # Loop over elements touching this node
    for elem_info in map.node_to_elements[node_i]
        elem = elements[elem_info.element_id]
        local_idx = elem_info.local_node_idx
        
        # Compute element contribution to node_i
        # (loop over integration points inside)
        compute_element_contribution!(contrib, elem, local_idx, u, time)
    end
    
    # Matrix-free matvec: w_i = K_i * u
    w[node_i] = matrix_vector_product_nodal(contrib, u)
end
```

## Comparison to Element Assembly

| Aspect | Element Assembly | Nodal Assembly |
|--------|------------------|----------------|
| **Outer loop** | Elements | Nodes |
| **Parallelization** | Element → atomics | Node → no atomics |
| **Storage** | Full K matrix (sparse) | 3×3 blocks per spider |
| **Matrix-free** | Difficult | Natural |
| **Contact** | Mismatch | Natural fit |
| **GPU** | Slow (atomics) | Fast (no atomics) |

## Connection to Golden Standard

This implements the architecture from `docs/src/book/multigpu_nodal_assembly.md`:

1. ✅ **Nodal assembly** (not element assembly)
2. ✅ **3×3 blocks** using `Tensor{2,3}` from Tensors.jl
3. ✅ **Matrix-free** matvec with spider pattern
4. ✅ **Zero allocations** (immutable Tensor types)

**Next steps:**

- Implement `compute_element_contribution!()` for real elements
- Integration with material models (already done: `compute_stress()` returns `SymmetricTensor{2,3}`)
- GPU kernels for nodal loop
- Contact mechanics integration

## Performance Implications

**2×2×2 Hex8 mesh (27 nodes, 81 DOFs):**

- **Element assembly:** 8 elements, each writes to overlapping DOFs → atomics
- **Nodal assembly:** 27 nodes, independent writes → no atomics

**Spider statistics:**

- Corner node: 8 couplings → compute 8 × 3×3 = 72 entries
- Interior node: 27 couplings → compute 27 × 3×3 = 243 entries (all nodes!)
- Average node: ~12 couplings → compute 12 × 3×3 = 108 entries

**Memory:** No global K matrix, only local K_blocks per thread (reused).

## Testing

See `test/test_nodal_assembly_structures.jl` for working examples:

```bash
cd /home/juajukka/dev/JuliaFEM.jl
julia --project=. test/test_nodal_assembly_structures.jl
```

**Tests:**

- ✅ Inverse mapping construction
- ✅ Spider computation  
- ✅ Nodal contribution storage
- ✅ Matrix-free matvec
- ✅ Efficiency analysis (hex mesh)

## References

1. **Golden standard:** `docs/src/book/multigpu_nodal_assembly.md`
2. **ARCHITECTURE.md:** Nodal assembly motivation
3. **TECHNICAL_VISION.md:** Why matrix-free iterative solvers

## Status

- **Implementation:** Prototype complete ✅
- **Testing:** Basic tests passing ✅
- **Integration:** Not yet integrated with main JuliaFEM
- **Performance:** Not yet benchmarked
- **GPU:** Not yet implemented (but designed for it)

This is the foundation for the modern JuliaFEM architecture!
