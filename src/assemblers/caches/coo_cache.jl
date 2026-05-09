# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
COO (Coordinate format) cache for element-based 3D continuum assembly.

COO format stores sparse matrices as triplets `(I, J, V)` where
`I[k]`, `J[k]`, and `V[k]` give the row, column, and value of the k-th
entry. After assembly the triplets are converted to a sparse matrix via
`sparse(I, J, V)`.
"""

using SparseArrays
using Tensors
using ..JuliaFEM: AssemblyMaterialWorkspace, GlobalMaterialCache, create_global_material_cache

"""
    ContinuumCOOCache <: AbstractAssemblerCache

Pre-allocated workspace for the element-based COO assembler with the
3D continuum-mechanics kernels (`ContinuumKernel`, …).

The cache wraps a [`ContinuumElementCache`](@ref), a `GeometryCache`,
an `AssemblyMaterialWorkspace`, and a persistent
[`GlobalMaterialCache`](@ref) covering every (integration point,
element) pair, so the inner assembly loop is allocation-free.

The shorter alias `COOCache` is kept for backwards compatibility.

# Fields
- `I, J, V` — pre-allocated triplet arrays (max capacity)
- `f::Vector{Float64}` — global force vector
- `element_cache::ContinuumElementCache` — per-element workspace
- `geometry_cache::GeometryCache` — geometry workspace
- `material_workspace::AssemblyMaterialWorkspace` — per-element material scratch
- `global_material_cache::GlobalMaterialCache` — persistent material state
- `counter::Int` — current position in the triplet arrays
- `capacity::Int` — maximum triplet capacity
- `ndofs::Int` — total number of DOFs

# Zero-allocation usage
```julia
cache = ContinuumCOOCache(mesh, kernel)
JuliaFEM.reset!(cache)
assemble!(cache, assembler, kernel, mesh)
K, f = extract_system(cache)
```
"""
mutable struct ContinuumCOOCache{EC<:ContinuumElementCache,GC<:GeometryCache,MC<:AbstractAssemblyMaterialWorkspace,GMC<:GlobalMaterialCache,FieldType<:NamedTuple,StateType<:NamedTuple} <: AbstractAssemblerCache
    I::Vector{Int}
    J::Vector{Int}
    V::Vector{Float64}
    f::Vector{Float64}
    element_cache::EC
    geometry_cache::GC
    material_workspace::MC
    global_material_cache::GMC
    counter::Int
    capacity::Int
    ndofs::Int
    𝔻_vec_buffer::Vector{SymmetricTensor{4,3,Float64,36}}
end

"""
    COOCache

Backwards-compatible alias for [`ContinuumCOOCache`](@ref).
"""
const COOCache = ContinuumCOOCache

"""
    ContinuumCOOCache(mesh, kernel) -> ContinuumCOOCache

Create a pre-allocated continuum COO cache.

The triplet capacity is over-allocated by 20% to absorb irregular
meshes; the rest of the workspace is sized exactly from the mesh and
kernel.
"""
function ContinuumCOOCache(mesh::AbstractMesh, kernel::AbstractKernel)
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

    # Persistent material state for the whole mesh (one entry per (ip, elem)).
    # For stateless materials this is a Matrix of empty NamedTuples — cheap and
    # gives every assembler the same single code path for state read/write.
    global_material_cache = create_global_material_cache(kernel.material;
                                                         n_ips=max_nips,
                                                         n_elems=nelems)

    # Infer FieldType and StateType from material_workspace for type stability
    # material_workspace is AssemblyMaterialWorkspace{FieldType, StateType}
    WorkspaceType = typeof(material_workspace)
    if WorkspaceType <: AssemblyMaterialWorkspace
        FieldType = WorkspaceType.parameters[1]
        StateType = WorkspaceType.parameters[2]
    else
        FieldType = NamedTuple
        StateType = NamedTuple
    end

    𝔻_vec_buffer = Vector{SymmetricTensor{4,3,Float64,36}}(undef, max_nips)

    return ContinuumCOOCache{typeof(element_cache), typeof(geometry_cache),
                              typeof(material_workspace), typeof(global_material_cache),
                              FieldType, StateType}(
        I, J, V, f, element_cache, geometry_cache, material_workspace,
        global_material_cache, 0, estimated_triplets, ndofs, 𝔻_vec_buffer)
end

"""
    reset!(cache::ContinuumCOOCache)

Reset triplet counter and zero the previously written triplet slots and
the force vector. Allocation-free; reuses the existing arrays.
"""
function reset!(cache::ContinuumCOOCache)
    # Only zero entries that were actually written (faster than fill!).
    current = cache.counter
    if current > 0
        @views cache.I[1:current] .= 0
        @views cache.J[1:current] .= 0
        @views cache.V[1:current] .= 0
    end
    fill!(cache.f, 0.0)
    cache.counter = 0
    return nothing
end

"""
    extract_system(cache::ContinuumCOOCache) -> (K, f)

Build the global sparse stiffness matrix from the triplets accumulated
in `cache` and return it together with the force vector. Allocates the
sparse matrix exactly once per assembly call.
"""
function extract_system(cache::ContinuumCOOCache)
    n = cache.counter
    I = @view cache.I[1:n]
    J = @view cache.J[1:n]
    V = @view cache.V[1:n]
    K = sparse(I, J, V, cache.ndofs, cache.ndofs)
    return K, cache.f
end

"""
    create_cache(assembler::COOAssembler, mesh, kernel) -> ContinuumCOOCache

Convenience constructor that forwards to `ContinuumCOOCache(mesh, kernel)`.
"""
function create_cache(assembler::COOAssembler, mesh::AbstractMesh, kernel::AbstractKernel)
    return ContinuumCOOCache(mesh, kernel)
end
