# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
COO (Coordinate format) cache for element-based assembly.

COO format stores sparse matrices as triplets (I, J, V) where:
- I[k] = row index of k-th entry
- J[k] = column index of k-th entry  
- V[k] = value of k-th entry

After assembly, triplets are converted to sparse matrix using `sparse(I, J, V)`.

# Performance
- Fast assembly (no structure lookups)
- Slow sparse matrix construction (O(nnz log nnz) for sorting)
- Memory overhead (stores all triplets including duplicates)

# Use case
Good for problems where sparsity pattern changes (e.g., contact, topology optimization).
"""

using SparseArrays
using Tensors
using ..JuliaFEM: AssemblyMaterialWorkspace

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
- `geometry_cache::GeometryCache`: Geometry workspace
- `material_workspace::AssemblyMaterialWorkspace`: Assembly material workspace (per-element temporary)
- `counter::Ref{Int}`: Current position in triplet arrays
- `capacity::Int`: Maximum triplet capacity
- `ndofs::Int`: Total number of DOFs

# Zero-Allocation Usage

```julia
cache = COOCache(mesh, kernel)
fill!(cache)  # Reset counter, zero arrays
assemble!(cache, assembler, kernel, mesh)  # No allocations
K, f = extract_system(cache)  # Build sparse matrix
```
"""
mutable struct COOCache{EC<:ElementCache,MC<:AbstractMaterialStateCache,FieldType<:NamedTuple,StateType<:NamedTuple} <: AbstractAssemblerCache
    I::Vector{Int}                      # Row indices
    J::Vector{Int}                      # Column indices
    V::Vector{Float64}                  # Values
    f::Vector{Float64}                  # Force vector
    element_cache::EC                   # Element workspace (concrete type!)
    geometry_cache::GeometryCache       # Geometry workspace
    material_workspace::MC              # Assembly material workspace (concrete type!)
    counter::Int                        # Current triplet count (Int instead of Ref{Int} for zero-allocation access)
    capacity::Int                       # Maximum triplet capacity
    ndofs::Int                          # Total number of DOFs
    𝔻_vec_buffer::Vector{SymmetricTensor{4,3,Float64,36}}  # Pre-allocated buffer for tangent vector (zero-allocation extraction)
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

    # Create all caches
    element_cache = create_element_cache(mesh, kernel)

    # Get max integration points for geometry and material caches
    max_nips = length(element_cache.ips)
    geometry_cache = create_geometry_cache(nnodes_elem, max_nips)
    material_workspace = create_material_cache(kernel.material, max_nips)
    
    # Infer FieldType and StateType from material_workspace for type stability
    # material_workspace is AssemblyMaterialWorkspace{FieldType, StateType}
    WorkspaceType = typeof(material_workspace)
    if WorkspaceType <: AssemblyMaterialWorkspace
        FieldType = WorkspaceType.parameters[1]
        StateType = WorkspaceType.parameters[2]
    else
        # Fallback for other material cache types (shouldn't happen in practice)
        FieldType = NamedTuple
        StateType = NamedTuple
    end
    
    # Pre-allocate buffer for tangent vector extraction (zero-allocation)
    𝔻_vec_buffer = Vector{SymmetricTensor{4,3,Float64,36}}(undef, max_nips)

    return COOCache{typeof(element_cache), typeof(material_workspace), FieldType, StateType}(
        I, J, V, f, element_cache, geometry_cache, material_workspace,
        0, estimated_triplets, ndofs, 𝔻_vec_buffer)  # Use Int instead of Ref(0) for zero-allocation
end

"""
    reset!(cache::COOCache)

Reset COO cache for new assembly.

Zeros out triplet arrays and force vector, resets counter.
**Zero allocations** - reuses existing arrays.
"""
function reset!(cache::COOCache)
    # Only zero up to current counter position (faster than fill!)
    current = cache.counter  # Direct access (Int, not Ref{Int})
    if current > 0
        @views cache.I[1:current] .= 0
        @views cache.J[1:current] .= 0
        @views cache.V[1:current] .= 0
    end
    fill!(cache.f, 0.0)
    cache.counter = 0  # Direct assignment (Int, not Ref{Int})
    return nothing
end

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
    n = cache.counter  # Direct access (Int, not Ref{Int})
    I = @view cache.I[1:n]
    J = @view cache.J[1:n]
    V = @view cache.V[1:n]
    K = sparse(I, J, V, cache.ndofs, cache.ndofs)
    return K, cache.f
end

"""
    create_cache(assembler::COOAssembler, mesh::AbstractMesh, kernel::AbstractKernel) -> COOCache

Create pre-allocated cache for COO assembly.

Convenience function that wraps `COOCache(mesh, kernel)`.

# Arguments
- `assembler`: COO assembler
- `mesh`: Finite element mesh
- `kernel`: Domain kernel

# Returns
- Pre-allocated COO cache

# Example

```julia
cache = create_cache(COOAssembler(), mesh, kernel)
assemble!(cache, COOAssembler(), kernel, mesh)
K, f = extract_system(cache)
```
"""
function create_cache(assembler::COOAssembler, mesh::AbstractMesh, kernel::AbstractKernel)
    return COOCache(mesh, kernel)
end
