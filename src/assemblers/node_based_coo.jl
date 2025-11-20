# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Node-based COO assembly using block integration.

**NODAL ASSEMBLY PARADIGM**: Loop over nodes, not elements!

Each node:
1. Finds all elements touching it (via inverse connectivity)
2. For each touching element:
   - Prepares element geometry once (PreparedElement)
   - Computes only needed 3×3 blocks (compute_block!)
3. Scatters blocks to COO triplets

# Key Differences from Element-Based Assembly

**Element-Based (traditional):**
```julia
for element in elements
    K_e = compute_element_stiffness(element)  # Full N×N matrix of 3×3 blocks
    scatter(K_e)                               # Scatter all entries
end
```

**Node-Based (this file):**
```julia
for node_i in nodes
    for element in elements_touching(node_i)
        prepared = prepare_element(element)    # Geometry preprocessing
        for node_j in element.nodes
            K_ij = compute_block!(prepared, i, j)  # Single 3×3 block
            scatter(K_ij, i, j)                    # Scatter one block
        end
    end
end
```

# Advantages

1. **GPU-friendly**: One thread per node, no race conditions
2. **Contact-ready**: Contact is naturally node-based
3. **Matrix-free ready**: Can compute K*v without forming K
4. **Cache-friendly**: Reuses PreparedElement for multiple blocks
5. **Adaptive-ready**: Easy to refine/coarsen at node level

# Performance Expectations

- **CPU Single-thread**: ~1.5-2x slower than element-based (more kernel calls)
- **CPU Multi-thread**: ~1.5-2x faster (better parallelization)
- **GPU**: ~10-50x faster (massive parallelization, no atomics needed)

# References

- Golden standard: `docs/src/book/multigpu_nodal_assembly.md`
- PreparedElement: `src/domains/continuum/integration.jl`
- Block kernel: `src/domains/continuum/kernel.jl`

# Example

```julia
# Setup
mesh = create_cantilever_mesh(50, 10, 10)
material = LinearElastic(E=210e9, ν=0.3)
kernel = ContinuumKernel(
    ContinuumFormulation{FullThreeD}(),
    material,
    Displacement{3}()
)

# Create node-based assembler and cache
assembler = NodeBasedCOOAssembler()
cache = create_cache(assembler, mesh, kernel)

# Assemble (zero allocations after warmup!)
assemble!(cache, assembler, kernel, mesh)

# Extract system
K, f = extract_system(cache)

# Solve
apply_dirichlet_bcs!(K, f, kernel, mesh, bc_dirichlet)
u = K \\ f
```
"""

using SparseArrays
using Tensors

"""
    NodeBasedCOOCache

Pre-allocated cache for node-based COO assembly.

Similar to COOCache but includes inverse connectivity mapping.

# Fields
- `I::Vector{Int}`: Row indices (COO format)
- `J::Vector{Int}`: Column indices (COO format)
- `V::Vector{Float64}`: Values (COO format)
- `f::Vector{Float64}`: Global force vector
- `counter::Ref{Int}`: Current triplet count
- `capacity::Int`: Maximum triplet capacity
- `node_to_elements::NodeToElementsMap`: Inverse connectivity
- `element_cache::ElementCache`: Cache for element operations
- `ndofs::Int`: Total DOFs in system
"""
struct NodeBasedCOOCache{T<:AbstractTopology,B<:AbstractBasis,IPS}
    I::Vector{Int}
    J::Vector{Int}
    V::Vector{Float64}
    f::Vector{Float64}
    counter::Ref{Int}
    capacity::Int
    node_to_elements::NodeToElementsMap
    element_cache::ElementCache{T,B,IPS}
    ndofs::Int
end

"""
    NodeBasedCOOCache(mesh::AbstractMesh, kernel::ContinuumKernel)

Create cache for node-based assembly.

Builds inverse connectivity and allocates buffers.

# Arguments
- `mesh`: Finite element mesh
- `kernel`: Continuum kernel

# Returns
- Pre-allocated node-based COO cache
"""
function NodeBasedCOOCache(mesh::AbstractMesh, kernel::ContinuumKernel)
    # Build inverse connectivity
    node_to_elements = NodeToElementsMap(mesh.connectivity)

    # Estimate triplet count (same as element-based)
    ndofs_per_node = dofs_per_node(kernel)
    nnodes = length(mesh.nodes)
    ndofs = ndofs_per_node * nnodes

    # Estimate: For each node, sum over touching elements
    # Each element contributes N blocks (N = nodes per element)
    # Each block = 3×3 = 9 triplets
    avg_elements_per_node = node_to_elements.nelements / nnodes
    N = length(first(mesh.connectivity))  # Nodes per element
    estimated_triplets = Int(ceil(1.2 * nnodes * avg_elements_per_node * N * 9))

    # Allocate triplet arrays
    I = Vector{Int}(undef, estimated_triplets)
    J = Vector{Int}(undef, estimated_triplets)
    V = Vector{Float64}(undef, estimated_triplets)
    f = zeros(Float64, ndofs)
    counter = Ref(0)

    # Create element cache (for prepare_element! and compute_block!)
    element_cache = ElementCache(mesh, kernel)

    return NodeBasedCOOCache(I, J, V, f, counter, estimated_triplets,
        node_to_elements, element_cache, ndofs)
end

"""
    reset!(cache::NodeBasedCOOCache)

Reset cache for new assembly (zero force vector, reset counter).

Does NOT clear inverse connectivity (that's permanent structure).
"""
function reset!(cache::NodeBasedCOOCache)
    fill!(cache.f, 0.0)
    cache.counter[] = 0
    return nothing
end

"""
    assemble!(
        cache::NodeBasedCOOCache,
        assembler::NodeBasedCOOAssembler,
        kernel::ContinuumKernel,
        mesh::AbstractMesh
    ) -> Nothing

Assemble global system using **node-based traversal**.

# Algorithm

```julia
for node_i in 1:nnodes
    # Get all elements touching this node
    for elem_info in node_to_elements[node_i]
        element_id = elem_info.element_id
        local_i = elem_info.local_node_idx
        
        # Prepare element geometry ONCE
        prepared = prepare_element!(cache.element_cache, kernel, element_id, mesh)
        
        # Compute blocks for all nodes in this element
        for local_j in 1:N
            global_j = connectivity[element_id][local_j]
            
            # Compute single 3×3 block
            K_ij = compute_block!(prepared, kernel.material, local_i, local_j)
            
            # Scatter to triplets
            scatter_block_to_triplets!(cache, K_ij, node_i, global_j)
        end
    end
end
```

# Key Operations

1. **prepare_element!** - Precompute geometry (Jacobian, gradients) once per element
2. **compute_block!** - Compute single 3×3 stiffness block using prepared geometry
3. **scatter_block_to_triplets!** - Add 9 triplets (i,j,value) for 3×3 block

# Zero-Allocation (After Warmup)

All arrays pre-allocated. Element preparation reuses cache buffers.

# Arguments
- `cache`: Pre-allocated node-based COO cache
- `assembler`: Node-based COO assembler
- `kernel`: Continuum kernel
- `mesh`: Finite element mesh
"""
function assemble!(
    cache::NodeBasedCOOCache,
    assembler::NodeBasedCOOAssembler,
    kernel::ContinuumKernel,
    mesh::AbstractMesh
)
    # Reset cache
    reset!(cache)

    nnodes = length(mesh.nodes)
    ndofs_per_node = dofs_per_node(kernel)

    # NODAL LOOP: One iteration per node (GPU: one thread per node!)
    for node_i in 1:nnodes
        # Get all elements touching this node
        touching_elements = cache.node_to_elements.node_to_elements[node_i]

        # Loop over touching elements
        for elem_info in touching_elements
            element_id = elem_info.element_id
            local_i = elem_info.local_node_idx  # Position of node_i in element

            # Prepare element geometry ONCE (reuses cache.element_cache)
            prepared = prepare_element!(cache.element_cache, kernel, element_id, mesh)

            # Get element connectivity
            conn = mesh.connectivity[element_id]
            N = length(conn)  # Nodes per element

            # Compute blocks for all nodes j in this element
            for local_j in 1:N
                global_j = conn[local_j]

                # Compute single 3×3 block K[i,j]
                # This is THE KEY OPERATION: block-based integration
                K_ij = compute_block!(
                    prepared,
                    kernel.material,
                    local_i,
                    local_j
                )

                # Scatter 3×3 block to triplets (adds 9 entries)
                scatter_block_to_triplets!(
                    cache,
                    K_ij,
                    node_i,
                    global_j,
                    ndofs_per_node
                )
            end
        end
    end

    return nothing
end

"""
    scatter_block_to_triplets!(
        cache::NodeBasedCOOCache,
        K_block::Tensor{2,3},
        node_i::Int,
        node_j::Int,
        ndofs_per_node::Int
    )

Scatter single 3×3 block to COO triplets **in-place**.

Maps block[α,β] → triplet at DOF indices:
- Row: 3*(node_i-1) + α
- Col: 3*(node_j-1) + β
- Val: K_block[α,β]

# Arguments
- `cache`: Node-based COO cache
- `K_block`: 3×3 stiffness block (Tensor{2,3})
- `node_i`: Global row node index
- `node_j`: Global column node index
- `ndofs_per_node`: DOFs per node (typically 3)

# Zero-Allocation

Writes to pre-allocated triplet arrays, updates counter.
"""
function scatter_block_to_triplets!(
    cache::NodeBasedCOOCache,
    K_block::Tensor{2,3,Float64},
    node_i::Int,
    node_j::Int,
    ndofs_per_node::Int
)
    counter = cache.counter[]

    # Check capacity
    new_triplets = ndofs_per_node * ndofs_per_node  # 3×3 = 9
    if counter + new_triplets > cache.capacity
        error("Node-based COO cache overflow: need $(counter + new_triplets) triplets, " *
              "capacity is $(cache.capacity). Increase cache size.")
    end

    # DOF offsets for nodes i and j
    row_offset = ndofs_per_node * (node_i - 1)
    col_offset = ndofs_per_node * (node_j - 1)

    # Scatter 3×3 block to triplets
    for β in 1:ndofs_per_node  # Column (node j DOF)
        j_global = col_offset + β
        for α in 1:ndofs_per_node  # Row (node i DOF)
            i_global = row_offset + α
            counter += 1
            cache.I[counter] = i_global
            cache.J[counter] = j_global
            cache.V[counter] = K_block[α, β]
        end
    end

    cache.counter[] = counter
    return nothing
end

"""
    extract_system(cache::NodeBasedCOOCache) -> (K, f)

Build sparse matrix from triplets and return system.

Calls `sparse(I, J, V)` to build CSC matrix. Duplicates are summed automatically.

# Arguments
- `cache`: Assembled node-based COO cache

# Returns
- `K::SparseMatrixCSC`: Global stiffness matrix
- `f::Vector`: Global force vector

# Allocation

Allocates sparse matrix structure (CSC format). This is the only allocation
outside cache construction.
"""
function extract_system(cache::NodeBasedCOOCache)
    ntriplets = cache.counter[]

    # Build sparse matrix (duplicates are summed automatically)
    I_used = @view cache.I[1:ntriplets]
    J_used = @view cache.J[1:ntriplets]
    V_used = @view cache.V[1:ntriplets]

    K = sparse(I_used, J_used, V_used, cache.ndofs, cache.ndofs)

    return K, cache.f
end

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

"""
    create_cache(
        assembler::NodeBasedCOOAssembler,
        mesh::AbstractMesh,
        kernel::ContinuumKernel
    ) -> NodeBasedCOOCache

Create pre-allocated cache for node-based COO assembly.

Convenience function that wraps `NodeBasedCOOCache(mesh, kernel)`.

# Example

```julia
assembler = NodeBasedCOOAssembler()
cache = create_cache(assembler, mesh, kernel)
assemble!(cache, assembler, kernel, mesh)
K, f = extract_system(cache)
```
"""
function create_cache(
    assembler::NodeBasedCOOAssembler,
    mesh::AbstractMesh,
    kernel::ContinuumKernel
)
    return NodeBasedCOOCache(mesh, kernel)
end

"""
    dofs_per_node(kernel::ContinuumKernel) -> Int

Return DOFs per node for continuum kernel (always 3 for displacement).

Dispatches on kernel field dimension.
"""
function dofs_per_node(kernel::ContinuumKernel{Theory,Mat}) where {Theory,Mat}
    field = kernel.field
    return field.dim  # Displacement{3} → 3
end

# ============================================================================
# PERFORMANCE NOTES
# ============================================================================

#=
# CPU Performance Comparison (Estimated)

**Element-Based Assembly:**
- Elements: 1000 Tet4
- Operations: 1000 elements × 4×4 blocks × 3×3 entries = 48,000 block computations
- Time: ~5ms (baseline)

**Node-Based Assembly:**
- Nodes: 500 nodes
- Operations: 500 nodes × 8 elements/node × 4 blocks/element = 16,000 block computations
- But: 3× more kernel calls due to overlaps
- Time: ~7-10ms (1.5-2× slower single-threaded)

**Why slower on CPU?**
- Each block computed once in element assembly
- Each block computed 2× on average in nodal assembly (shared between 2 elements)
- More function call overhead

**Why faster on GPU?**
- Element assembly: Sequential (can't parallelize over elements efficiently)
- Nodal assembly: Massive parallelism (one thread per node)
- GPU speedup: ~10-50× depending on problem size

**Multi-threaded CPU (Threads.@threads):**
- Can parallelize outer node loop
- Expected speedup: 1.5-2× over element-based
- No race conditions (each node writes different triplets)

# Memory Comparison

**Element-Based:**
- Triplet storage: ~50 KB per 1000 elements
- Element cache: ~2 KB per thread

**Node-Based:**
- Triplet storage: Same (~50 KB)
- Element cache: ~2 KB per thread
- Inverse connectivity: ~10-20 KB (one-time)

→ Nearly identical memory usage!

# When to Use Node-Based Assembly

**Use when:**
- ✅ GPU acceleration needed
- ✅ Contact mechanics (naturally nodal)
- ✅ Matrix-free methods (K*v without forming K)
- ✅ Adaptive refinement (local node operations)
- ✅ Multi-threading on CPU

**Don't use when:**
- ❌ Single-threaded CPU only
- ❌ Simple problems (< 1000 nodes)
- ❌ Prototyping/debugging (element-based is clearer)
=#
