# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
COO (Coordinate format) assembly implementation.

Classical element-by-element assembly using triplet vectors (I, J, V).
Accumulates all element contributions, builds sparse matrix at end.

Performance: Baseline (1.0x), moderate memory usage.
Best for: Prototyping, debugging, simple problems.
"""

using SparseArrays
using ..JuliaFEM: GlobalMaterialCache, get_tangent, get_tangent_vector, extract_tangent!

include("scatter_blocks_to_triplets_symmetric_direct.jl")
include("scatter_blocks_to_force.jl")

"""
    assemble_element!(
        element_cache, geometry_cache, material_workspace,
        kernel, elem_id, mesh, N, u_global, global_cache, Δt,
        𝔻_vec_buffer,
    ) -> Nothing

Assemble a single element using `GlobalMaterialCache` for persistent state.

The element pipeline runs as four in-place phases on pre-allocated caches:

1. `reset!` element / geometry / material workspaces.
2. `update_element_cache!` extracts displacements and the DOF mapping.
3. `update_geometry_cache!` extracts node coordinates and computes
   physical gradients and `detJ * w`.
4. `update_material_cache!` reads the old state from `global_cache`,
   computes stress and tangent, and writes the new state back.
5. The K block contributions are integrated against the tangent buffer.

The function should be zero-allocation in the inner assembly loop; all
working storage lives on the caches passed in.
"""
function assemble_element!(
    element_cache::ElementCache,
    geometry_cache::GeometryCache,
    material_workspace::AssemblyMaterialWorkspace{FieldType, StateType},
    kernel::AbstractKernel,
    elem_id::Int,
    mesh::AbstractMesh,
    N::Int,
    u_global::Union{Nothing,Vector{Vec{3,Float64}}},
    global_cache::GlobalMaterialCache,
    Δt::Float64,
    𝔻_vec_buffer::Vector{SymmetricTensor{4,3,Float64,36}}
) where {FieldType<:NamedTuple, StateType<:NamedTuple}
    reset!(element_cache)
    reset!(geometry_cache)
    reset!(material_workspace)

    update_element_cache!(element_cache, kernel, elem_id, mesh, u_global)
    update_geometry_cache!(geometry_cache, element_cache, elem_id, mesh)
    update_material_cache!(material_workspace, geometry_cache, kernel.material,
                           element_cache, global_cache, elem_id, Δt)

    # Extract tangent vector once before the integration loop using a
    # pre-allocated buffer (compile-time field index keeps it allocation-free).
    fields = getfield(material_workspace, 1)
    extract_tangent!(𝔻_vec_buffer, fields, FieldType)
    𝔻_vec = 𝔻_vec_buffer
    @inbounds for k in 1:N, l in k:N  # upper triangle only — stiffness is symmetric
        compute_block!(
            element_cache.K_blocks,
            geometry_cache.∇N_data,
            geometry_cache.detJ_w,
            𝔻_vec,
            k, l,
        )
    end

    return nothing
end

"""
    assemble!(cache, assembler, kernel, mesh,
              u_global, global_cache, Δt) -> Nothing
    assemble!(cache, assembler, kernel, mesh,
              u_global = nothing, Δt::Float64 = 0.0) -> Nothing

Assemble stiffness matrix and force vector for an element-based COO sweep.

The element loop drives `assemble_element!` and scatters block contributions
into the cache's pre-allocated triplet arrays. Material state is owned by
`global_cache`; if the convenience overload is used, the cache's embedded
`global_material_cache` (built once in the constructor) is reused.

# Arguments
- `cache`: pre-allocated `COOCache`.
- `assembler`: `COOAssembler` instance.
- `kernel`: domain kernel (continuum / heat / mixed / ...).
- `mesh`: finite element mesh.
- `u_global`: global displacement field, or `nothing` for linear analysis.
- `global_cache`: persistent material state container.
- `Δt`: time increment.

# Side effects
- Mutates `cache.I`, `cache.J`, `cache.V`, and `cache.f`.
- Writes new states back to `global_cache` via `set_state!`.

# Zero-allocation
No allocations occur during the inner loop. The only allocation in the
typical workflow is `sparse(I, J, V)` inside `extract_system(cache)`.
"""
function assemble!(
    cache::COOCache,
    assembler::COOAssembler,
    kernel::AbstractKernel,
    mesh::AbstractMesh,
    u_global::Union{Nothing,Vector{Vec{3,Float64}}},
    global_cache::GlobalMaterialCache,
    Δt::Float64,
)
    MeshType = typeof(mesh)
    N = MeshType.parameters[1]::Int  # nodes per element — compile-time constant

    reset!(cache)

    nelems = nelements(mesh)
    element_cache = cache.element_cache
    geometry_cache = cache.geometry_cache
    material_workspace = cache.material_workspace
    𝔻_vec_buffer = cache.𝔻_vec_buffer
    counter = 0

    for elem_id in 1:nelems
        assemble_element!(element_cache, geometry_cache, material_workspace,
            kernel, elem_id, mesh, N, u_global, global_cache, Δt, 𝔻_vec_buffer)

        counter = scatter_blocks_to_triplets_symmetric_direct!(
            cache.I, cache.J, cache.V, counter, cache.capacity,
            element_cache.K_blocks, element_cache.dofs, N)

        scatter_blocks_to_force!(cache.f, element_cache.f_blocks, element_cache.dofs, N)
    end

    cache.counter = counter
    return nothing
end

# Convenience entry point: reuse the cache's embedded global_material_cache
# so callers in linear analyses do not have to build one explicitly.
function assemble!(
    cache::COOCache,
    assembler::COOAssembler,
    kernel::AbstractKernel,
    mesh::AbstractMesh,
    u_global::Union{Nothing,Vector{Vec{3,Float64}}} = nothing,
    Δt::Float64 = 0.0,
)
    return assemble!(cache, assembler, kernel, mesh,
                     u_global, cache.global_material_cache, Δt)
end

"""
    estimate_triplet_count(mesh::AbstractMesh, kernel::AbstractKernel) -> Int

Estimate number of triplets for COO assembly.

Used to pre-allocate triplet arrays with correct capacity.

# Formula

```
triplet_count = sum over elements of ndofs_elem^2
              ≈ nelems × (avg_nnodes_per_elem × ndofs_per_node)^2
```

Over-allocates by 20% for irregular meshes.

# Arguments
- `mesh`: Finite element mesh
- `kernel`: Domain kernel

# Returns
- Estimated triplet count (integer)
"""
function estimate_triplet_count(mesh::AbstractMesh, kernel::AbstractKernel)
    nelems = nelements(mesh)
    ndofs_per_node = dofs_per_node(kernel)

    # Compute average nodes per element
    avg_nnodes_per_elem = sum(nnodes_per_element(mesh, i) for i in 1:nelems) / nelems
    avg_ndofs_per_elem = Int(ceil(avg_nnodes_per_elem * ndofs_per_node))

    # Estimate: nelems * ndofs_elem^2, with 20% safety margin
    estimated = Int(ceil(1.2 * nelems * avg_ndofs_per_elem^2))

    return estimated
end
