# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
COO (Coordinate format) assembly implementation.

Classical element-by-element assembly using triplet vectors (I, J, V).
Accumulates all element contributions, builds sparse matrix at end.

**Performance**: Baseline (1.0x), moderate memory usage.
**Best for**: Prototyping, debugging, simple problems.
"""

using SparseArrays

# Include scatter implementations
include("scatter_to_triplets.jl")
include("scatter_blocks_to_triplets.jl")
include("scatter_blocks_to_triplets_symmetric.jl")
include("scatter_blocks_to_triplets_symmetric_manually_unrolled.jl")
include("scatter_blocks_to_triplets_symmetric_direct.jl")
include("scatter_to_force.jl")
include("scatter_blocks_to_force.jl")

"""
    assemble_element!(
        element_cache::ElementCache,
        geometry_cache::GeometryCache,
        material_cache::MaterialStateCache,
        kernel::AbstractKernel,
        elem_id::Int,
        mesh::AbstractMesh,
        N::Int,
        u_global::Union{Nothing,Vector{Vec{3,Float64}}},
        state_old::Union{Nothing,Matrix{<:AbstractMaterialState}},
        Δt::Float64
    ) -> Nothing

Assemble a single element (inline function for benchmarking).

This function encapsulates all operations performed on a single element:
1. Reset caches
2. Update element cache (extract displacements, DOF mapping)
3. Update geometry cache (extract coordinates, compute gradients, detJ*w)
4. Update material cache (compute stress, tangent, internal state)
5. Compute element stiffness blocks

# Arguments
- `element_cache`: Element cache to update
- `geometry_cache`: Geometry cache to update
- `material_cache`: Material cache to update
- `kernel`: Domain kernel
- `elem_id`: Current element ID
- `mesh`: Finite element mesh
- `N`: Number of nodes per element (compile-time constant)
- `u_global`: Global displacement field (nothing for linear analysis)
- `state_old`: Global material state (nothing for stateless materials)
- `Δt`: Time increment

# Zero-Allocation Guarantee
This function should have ZERO allocations when called in a loop.
All operations are in-place mutations of pre-allocated caches.
"""
function assemble_element!(
    element_cache::ElementCache,
    geometry_cache::GeometryCache,
    material_cache::MaterialStateCache,
    kernel::AbstractKernel,
    elem_id::Int,
    mesh::AbstractMesh,
    N::Int,
    u_global::Union{Nothing,Vector{Vec{3,Float64}}},
    state_old::Union{Nothing,Matrix{<:AbstractMaterialState}},
    Δt::Float64
)
    # Reset caches for new element
    reset!(element_cache)
    reset!(geometry_cache)
    reset!(material_cache)

    # PHASE 1: Update element cache (extract displacements, DOF mapping)
    update_element_cache!(element_cache, kernel, elem_id, mesh, u_global)

    # PHASE 2: Update geometry cache (extract coordinates, compute gradients, detJ*w)
    update_geometry_cache!(geometry_cache, element_cache, kernel, elem_id, mesh)

    # PHASE 3: Update material cache (compute stress, tangent, internal state)
    update_material_cache!(material_cache, geometry_cache, kernel.material, element_cache, state_old, elem_id, Δt)

    # PHASE 4: Compute element stiffness blocks
    # Assemble only upper triangle (k ≤ l) since stiffness matrix is symmetric
    # This halves computation and memory usage
    @inbounds for k in 1:N, l in k:N  # Only l ≥ k (upper triangle)
        compute_block!(
            element_cache.K_blocks,
            geometry_cache.∇N_data,
            geometry_cache.detJ_w,
            material_cache.𝔻,
            k, l
        )
    end

    return nothing
end

"""
    assemble!(
        cache::COOCache,
        assembler::COOAssembler,
        kernel::AbstractKernel,
        mesh::AbstractMesh,
        u_global::Union{Nothing,Vector{Vec{3,Float64}}} = nothing,
        state_old::Union{Nothing,Matrix{<:AbstractMaterialState}} = nothing,
        Δt::Float64 = 0.0
    ) -> Nothing

Assemble global system using COO format with **three-phase approach**.

# Three-Phase Algorithm

1. Reset cache (zero arrays, reset counter)
2. Loop over elements:
   a. Reset caches: `reset!(geometry_cache)`, `reset!(element_cache)`, `reset!(material_cache)`
   b. **Phase 1a (Geometry):** Extract node coordinates
      - `update_geometry_cache!(geometry_cache, kernel, elem_id, mesh)`
   c. **Phase 1b (Element):** Extract displacements and DOF mapping
      - `update_element_cache!(element_cache, kernel, elem_id, mesh, u_global)`
   d. **Phase 2 (Material):** Compute material state at all IPs
      - `update_material_cache!(material_cache, geometry_cache, material, element_cache, state_old, elem_id, Δt)`
   e. **Phase 3 (Stiffness):** Compute element stiffness using precomputed state
      - `compute_element_stiffness!(element_cache, geometry_cache, material_cache, N, NIP)`
   f. Scatter Ke to triplets: accumulate (i,j,value) to (I,J,V)
   g. Scatter fe to global force vector: `f[dofs] += fe`
3. Use `extract_system(cache)` to build sparse matrix from triplets

# Arguments
- `cache`: Pre-allocated COO cache
- `assembler`: COO assembler
- `kernel`: Domain kernel (continuum, plate, beam, etc.)
- `mesh`: Finite element mesh
- `u_global`: Global displacement field [nnodes] as Vec{3} (nothing for linear analysis)
- `state_old`: Global material state [nips, nelems] (nothing for stateless materials)
- `Δt`: Time increment (for rate-dependent materials)

# Zero-Allocation Guarantee

No allocations during assembly loop. All arrays pre-allocated in cache.
Only allocation: `sparse(I, J, V)` in `extract_system(cache)` (called once).

# Example

    # Setup (one-time)
    mesh = create_cantilever_mesh(10, 2, 2)
    kernel = ContinuumKernel(formulation, material, field)
    assembler = COOAssembler()
    cache = COOCache(mesh, kernel)

    # Linear assembly (no displacement, no state)
    assemble!(cache, assembler, kernel, mesh)

    # Nonlinear assembly (with displacement and state)
    nnodes = nnodes_total(mesh)
    u_global = [zero(Vec{3,Float64}) for _ in 1:nnodes]
    nips = length(cache.element_cache.ips)
    nelems = nelements(mesh)
    state_old = Matrix{PlasticityState}(undef, nips, nelems)
    for i in 1:nips, j in 1:nelems
        state_old[i,j] = PlasticityState()  # Initialize with zero state
    end

    for iter in 1:max_iter
        assemble!(cache, assembler, kernel, mesh, u_global, state_old, Δt)
        K, f = extract_system(cache)
        # ... solve, update u_global and state_old ...
    end

# Performance

For 2500 Tet4 elements:
- Time: 9.71 ms
- Memory: 8.4 MB
- Speedup: 1.0x (baseline)
"""
function assemble!(
    cache::COOCache,
    assembler::COOAssembler,
    kernel::AbstractKernel,
    mesh::AbstractMesh,
    u_global::Union{Nothing,Vector{Vec{3,Float64}}}=nothing,
    state_old::Union{Nothing,Matrix{<:AbstractMaterialState}}=nothing,
    Δt::Float64=0.0
)
    # Extract compile-time constants from mesh type parameters
    # Mesh{N,T} where N = nodes per element, T = topology type
    MeshType = typeof(mesh)
    N = MeshType.parameters[1]::Int  # Compile-time constant for loop unrolling

    # Reset cache for new assembly
    reset!(cache)

    nelems = nelements(mesh)
    element_cache = cache.element_cache
    geometry_cache = cache.geometry_cache
    material_cache = cache.material_cache

    # N (nodes per element) and NIP (integration points) are now compile-time constants
    # extracted from type parameters for aggressive loop unrolling

    # Extract counter ONCE before loop to avoid Ref{Int} indirection overhead
    #counter = cache.counter[]
    counter = 0

    # Loop over elements
    for elem_id in 1:nelems
        # Assemble single element (all phases)
        assemble_element!(element_cache, geometry_cache, material_cache,
            kernel, elem_id, mesh, N, u_global, state_old, Δt)

        # Scatter blocks directly to triplets using direct version (zero dispatch!)
        # Pass counter as Int (not Ref{Int}) to eliminate indirection
        counter = scatter_blocks_to_triplets_symmetric_direct!(
            cache.I, cache.J, cache.V, counter, cache.capacity,
            element_cache.K_blocks, element_cache.dofs, N)

        # Scatter blocked force to global force vector
        scatter_blocks_to_force!(cache.f, element_cache.f_blocks, element_cache.dofs, N)
    end

    # Write counter back ONCE after loop
    # NOTE: counter MUST be written back so extract_system() knows how many triplets to extract
    cache.counter[] = counter

    return nothing
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
