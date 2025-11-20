# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Nodal cache for node-based assembly.

Nodal assembly iterates over nodes rather than elements. Each node assembles
contributions from all touching elements. This approach has advantages for:
- Contact mechanics (contact is inherently nodal)
- Domain decomposition (clear node ownership)
- Matrix-free operations (natural node-based matvec)
- Adaptive refinement (local node operations)

# Structure
Pre-builds node-to-elements map (inverse connectivity) once. During assembly,
each node visits all touching elements and accumulates contributions.

# Performance
Similar to CSC for standard problems. Better locality for nodal operations
like contact and matrix-free solvers.

# Use case
Best for problems with nodal phenomena (contact, nodal plasticity) or
matrix-free iterative solvers.
"""

using SparseArrays

"""
    NodalCache <: AbstractAssemblerCache

Cache for nodal-based assembly.

Pre-allocates sparse matrix, force vector, and node-to-elements map.
Each node assembles contributions from all touching elements.

# Fields
- `K::SparseMatrixCSC{Float64,Int}`: Sparse matrix
- `f::Vector{Float64}`: Global force vector
- `node_cache::NodeCache`: Per-node workspace
- `element_cache::ElementCache`: Per-element workspace (for kernel calls)
- `node_to_elements::NodeToElementsMap`: Inverse connectivity

# Zero-Allocation Usage

```julia
cache = NodalCache(mesh, kernel)
fill!(cache)
assemble!(cache, assembler, kernel, mesh)  # No allocations
K, f = extract_system(cache)
```
"""
mutable struct NodalCache <: AbstractAssemblerCache
    K::SparseMatrixCSC{Float64,Int}  # Sparse matrix
    f::Vector{Float64}                # Force vector
    node_cache::NodeCache             # Node workspace
    element_cache::ElementCache       # Element workspace
    node_to_elements::NodeToElementsMap  # Inverse connectivity
end

"""
    NodalCache(mesh::AbstractMesh, kernel::AbstractKernel) -> NodalCache

Create pre-allocated nodal cache.

Builds node-to-elements map (inverse connectivity) for efficient nodal traversal.

# Arguments
- `mesh`: Finite element mesh
- `kernel`: Domain kernel defining DOF structure

# Returns
- `NodalCache` with pre-allocated workspace and inverse connectivity
"""
function NodalCache(mesh::AbstractMesh, kernel::AbstractKernel)
    ndofs_per_node = dofs_per_node(kernel)
    nnodes_mesh = nnodes_total(mesh)
    ndofs = nnodes_mesh * ndofs_per_node

    # Build sparsity pattern (same as CSC)
    K = build_sparsity_pattern(mesh, kernel)
    f = zeros(Float64, ndofs)

    # Create caches
    node_cache = create_node_cache(mesh, kernel)
    element_cache = create_element_cache(mesh, kernel)

    # Build inverse connectivity
    node_to_elements = NodeToElementsMap(mesh)

    return NodalCache(K, f, node_cache, element_cache, node_to_elements)
end

"""
    reset!(cache::NodalCache)

Reset nodal cache for new assembly.

Zeros out matrix values and force vector.
**Zero allocations** - reuses existing arrays.
"""
function reset!(cache::NodalCache)
    fill!(cache.K.nzval, 0.0)
    fill!(cache.f, 0.0)
    return nothing
end

"""
    extract_system(cache::NodalCache) -> (K, f)

Extract global system from nodal cache.

Returns references to sparse matrix and force vector.
**Zero allocations** - no copying.

# Arguments
- `cache`: Nodal cache after assembly

# Returns
- `K`: Sparse matrix (reference, no copy)
- `f`: Force vector (reference, no copy)
"""
function extract_system(cache::NodalCache)
    return cache.K, cache.f
end
