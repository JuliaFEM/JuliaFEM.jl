# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Cache structures for zero-allocation assembly.

All assemblers use pre-allocated caches containing:
- Global system matrices and vectors (K, f)
- Element/node-level workspace
- Sparse matrix structures

Caches are created once per problem and reused across multiple assembly
calls (e.g., in nonlinear iterations or time stepping).
"""

using SparseArrays

"""
    COOCache <: AbstractAssemblerCache

Cache for COO (coordinate format) assembly.

Pre-allocates triplet vectors `(I, J, V)` and workspace for element assembly.
After assembly, triplets are converted to sparse matrix using `sparse(I, J, V)`.

# Fields
- `I::Vector{Int}`: Row indices (pre-allocated, max capacity)
- `J::Vector{Int}`: Column indices (pre-allocated, max capacity)
- `V::Vector{Float64}`: Values (pre-allocated, max capacity)
- `f::Vector{Float64}`: Global force vector
- `element_cache::ElementCache`: Per-element workspace
- `counter::Ref{Int}`: Current position in triplet arrays

# Zero-Allocation Usage

```julia
cache = COOCache(mesh, kernel)
fill!(cache)  # Reset counter, zero arrays
assemble!(cache, assembler, kernel, mesh)  # No allocations
K, f = extract_system(cache)  # Build sparse matrix
```
"""
mutable struct COOCache <: AbstractAssemblerCache
    I::Vector{Int}              # Row indices
    J::Vector{Int}              # Column indices
    V::Vector{Float64}          # Values
    f::Vector{Float64}          # Force vector
    element_cache::ElementCache # Element workspace
    counter::Ref{Int}           # Current triplet count
    capacity::Int               # Maximum triplet capacity
    ndofs::Int                  # Total number of DOFs
end

"""
    COOCache(mesh, kernel) -> COOCache

Create pre-allocated COO cache.

Estimates maximum triplet count based on mesh connectivity and DOF structure.
Over-allocates by 20% to handle irregular meshes safely.

# Arguments
- `mesh`: Finite element mesh
- `kernel`: Domain kernel defining DOF structure

# Returns
- `COOCache` with pre-allocated triplet arrays
"""
function COOCache(mesh::AbstractMesh, kernel::AbstractKernel)
    nelems = nelements(mesh)
    ndofs_per_node = dofs_per_node(kernel)
    nnodes_total_mesh = nnodes_total(mesh)
    ndofs = nnodes_total_mesh * ndofs_per_node

    # Estimate triplet count: sum over elements of ndofs_elem^2
    # For uniform mesh: nelems * (nnodes_per_elem * ndofs_per_node)^2
    # Over-allocate by 20% for safety
    # For Mesh{N,T}, N is the first type parameter (nnodes_per_elem)
    MeshType = typeof(mesh)
    nnodes_elem = MeshType.parameters[1]::Int
    avg_ndofs_per_elem = Int(ceil(nnodes_elem * ndofs_per_node))
    estimated_triplets = Int(ceil(1.2 * nelems * avg_ndofs_per_elem^2))

    I = zeros(Int, estimated_triplets)
    J = zeros(Int, estimated_triplets)
    V = zeros(Float64, estimated_triplets)
    f = zeros(Float64, ndofs)
    element_cache = create_element_cache(mesh, kernel)

    return COOCache(I, J, V, f, element_cache, Ref(0), estimated_triplets, ndofs)
end

"""
    reset!(cache::COOCache)

Reset COO cache for new assembly.

Zeros out triplet arrays and force vector, resets counter.
**Zero allocations** - reuses existing arrays.
"""
function reset!(cache::COOCache)
    # Only zero up to current counter position (faster than fill!)
    current = cache.counter[]
    if current > 0
        @views cache.I[1:current] .= 0
        @views cache.J[1:current] .= 0
        @views cache.V[1:current] .= 0
    end
    fill!(cache.f, 0.0)
    cache.counter[] = 0
    return nothing
end

"""
    CSCCache <: AbstractAssemblerCache

Cache for CSC (compressed sparse column) assembly with pre-built structure.

Pre-allocates CSC sparse matrix with correct sparsity pattern. During assembly,
element contributions are merged directly into CSC arrays using two-pointer
algorithm. **4.1x faster than COO**, **16.6x less memory**.

# Fields
- `K::SparseMatrixCSC{Float64,Int}`: Sparse matrix with pre-built structure
- `f::Vector{Float64}`: Global force vector
- `element_cache::ElementCache`: Per-element workspace
- `colptr_cache::Vector{Int}`: Column pointer positions (for in-place merge)

# Zero-Allocation Usage

```julia
cache = CSCCache(mesh, kernel)  # Builds sparsity pattern (one-time cost)
fill!(cache)  # Zero values, keep structure
assemble!(cache, assembler, kernel, mesh)  # No allocations, in-place merge
K, f = extract_system(cache)  # Just returns references
```
"""
mutable struct CSCCache <: AbstractAssemblerCache
    K::SparseMatrixCSC{Float64,Int}  # Pre-built sparse matrix
    f::Vector{Float64}                # Force vector
    element_cache::ElementCache       # Element workspace
    colptr_cache::Vector{Int}         # Working column pointers
end

"""
    CSCCache(mesh::AbstractMesh, kernel::AbstractKernel) -> CSCCache

Create pre-allocated CSC cache with pre-built sparsity pattern.

Builds sparse matrix structure by:
1. Collecting all (i,j) pairs from element connectivity
2. Removing duplicates
3. Creating CSC structure with `sparse(I, J, zeros, m, n)`

Structure is reused across all subsequent assemblies (nonlinear iterations).

# Arguments
- `mesh`: Finite element mesh
- `kernel`: Domain kernel defining DOF structure

# Returns
- `CSCCache` with pre-built sparse matrix structure
"""
function CSCCache(mesh::AbstractMesh, kernel::AbstractKernel)
    ndofs_per_node = dofs_per_node(kernel)
    nnodes_mesh = nnodes_total(mesh)
    ndofs = nnodes_mesh * ndofs_per_node

    # Build sparsity pattern from mesh connectivity
    K = build_sparsity_pattern(mesh, kernel)
    f = zeros(Float64, ndofs)
    element_cache = create_element_cache(mesh, kernel)

    # Cache working column pointers (for two-pointer merge)
    colptr_cache = copy(K.colptr)

    return CSCCache(K, f, element_cache, colptr_cache)
end

"""
    build_sparsity_pattern(mesh::AbstractMesh, kernel::AbstractKernel) -> SparseMatrixCSC

Build sparse matrix structure from mesh connectivity.

Collects all (i,j) DOF pairs from element connectivity, creates CSC structure
with zero values. Structure is reused for all subsequent assemblies.

# Algorithm
1. Loop over elements
2. For each element, get DOF mapping
3. For all DOF pairs (i,j) in element, record (i,j)
4. Remove duplicates
5. Create `sparse(I, J, zeros, ndofs, ndofs)`

# Arguments
- `mesh`: Finite element mesh
- `kernel`: Domain kernel defining DOF structure

# Returns
- Sparse matrix with correct structure, zero values
"""
function build_sparsity_pattern(mesh::AbstractMesh, kernel::AbstractKernel)
    ndofs_per_node = dofs_per_node(kernel)
    nnodes_mesh = nnodes_total(mesh)
    ndofs = nnodes_mesh * ndofs_per_node
    nelems = nelements(mesh)

    # Estimate triplet count for pre-allocation
    MeshType = typeof(mesh)
    nnodes_elem = MeshType.parameters[1]::Int
    avg_ndofs_per_elem = Int(ceil(nnodes_elem * ndofs_per_node))
    estimated_triplets = Int(ceil(1.2 * nelems * avg_ndofs_per_elem^2))

    I = Vector{Int}()
    J = Vector{Int}()
    sizehint!(I, estimated_triplets)
    sizehint!(J, estimated_triplets)

    # Temporary DOF buffer
    dof_buffer = zeros(Int, avg_ndofs_per_elem)

    # Collect all (i,j) pairs from connectivity
    for elem_id in 1:nelems
        # Get element nodes
        nodes = mesh.connectivity[elem_id]
        nnodes_elem = length(nodes)
        ndofs_elem = nnodes_elem * ndofs_per_node

        # Get global DOF indices
        resize!(dof_buffer, ndofs_elem)
        get_dof_mapping!(dof_buffer, kernel, elem_id, mesh)

        # Record all (i,j) pairs
        for i_local in 1:ndofs_elem
            i_global = dof_buffer[i_local]
            for j_local in 1:ndofs_elem
                j_global = dof_buffer[j_local]
                push!(I, i_global)
                push!(J, j_global)
            end
        end
    end

    # Build CSC structure (sparse automatically removes duplicates)
    K = sparse(I, J, zeros(Float64, length(I)), ndofs, ndofs)

    return K
end

"""
    reset!(cache::CSCCache)

Reset CSC cache for new assembly.

Zeros out matrix values and force vector, keeps structure.
**Zero allocations** - reuses existing arrays.
"""
function reset!(cache::CSCCache)
    fill!(cache.K.nzval, 0.0)  # Zero values, keep structure
    fill!(cache.f, 0.0)
    copy!(cache.colptr_cache, cache.K.colptr)  # Reset column pointers
    return nothing
end

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

# Extract system from cache

"""
    extract_system(cache::COOCache) -> (K, f)

Extract global system from COO cache.

Builds sparse matrix from accumulated triplets. **Allocates** - only call
once per assembly.

# Arguments
- `cache`: COO cache after assembly

# Returns
- `K`: Sparse matrix built from triplets
- `f`: Force vector (reference, no copy)
"""
function extract_system(cache::COOCache)
    n = cache.counter[]
    I = @view cache.I[1:n]
    J = @view cache.J[1:n]
    V = @view cache.V[1:n]
    K = sparse(I, J, V, cache.ndofs, cache.ndofs)
    return K, cache.f
end

"""
    extract_system(cache::CSCCache) -> (K, f)

Extract global system from CSC cache.

Returns references to pre-built sparse matrix and force vector.
**Zero allocations** - no copying.

# Arguments
- `cache`: CSC cache after assembly

# Returns
- `K`: Sparse matrix (reference, no copy)
- `f`: Force vector (reference, no copy)
"""
function extract_system(cache::CSCCache)
    return cache.K, cache.f
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
