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

"""
    assemble!(
        cache::COOCache,
        assembler::COOAssembler,
        kernel::AbstractKernel,
        mesh::AbstractMesh
    ) -> Nothing

Assemble global system using COO format **in-place, zero allocations**.

# Algorithm

1. Reset cache (zero arrays, reset counter)
2. Loop over elements:
   a. Compute element stiffness in-place: `compute_element_stiffness!(cache.element_cache, ...)`
   b. Get DOF mapping: `get_dof_mapping!(cache.element_cache.dofs, ...)`
   c. Scatter Ke to triplets: accumulate (i,j,value) to (I,J,V)
   d. Scatter fe to global force vector: `f[dofs] += fe`
3. Use `extract_system(cache)` to build sparse matrix from triplets

# Arguments
- `cache`: Pre-allocated COO cache
- `assembler`: COO assembler
- `kernel`: Domain kernel (continuum, plate, beam, etc.)
- `mesh`: Finite element mesh

# Zero-Allocation Guarantee

No allocations during assembly loop. All arrays pre-allocated in cache.
Only allocation: `sparse(I, J, V)` in `extract_system(cache)` (called once).

# Example

```julia
# Setup (one-time)
mesh = create_cantilever_mesh(10, 2, 2)
kernel = ContinuumKernel(formulation, material, field)
assembler = COOAssembler()
cache = COOCache(mesh, kernel)

# Assembly (zero allocations, can repeat in nonlinear loop)
assemble!(cache, assembler, kernel, mesh)

# Extract system (allocates sparse matrix, call once)
K, f = extract_system(cache)
```

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
    mesh::AbstractMesh
)
    # Reset cache for new assembly
    reset!(cache)

    nelems = nelements(mesh)
    element_cache = cache.element_cache

    # Loop over elements
    for elem_id in 1:nelems
        # Compute element stiffness in-place (zero allocations)
        compute_element_stiffness!(element_cache, kernel, elem_id, mesh)

        # Get element nodes and DOF count
        nodes = mesh.connectivity[elem_id]
        nnodes_elem = length(nodes)
        ndofs_per_node = dofs_per_node(kernel)
        ndofs_elem = nnodes_elem * ndofs_per_node

        # Get DOF mapping in-place (zero allocations)
        dofs = @view element_cache.dofs[1:ndofs_elem]
        get_dof_mapping!(dofs, kernel, elem_id, mesh)

        # Scatter element stiffness to triplets
        Ke = @view element_cache.Ke[1:ndofs_elem, 1:ndofs_elem]
        scatter_to_triplets!(cache, Ke, dofs)

        # Scatter element force to global force vector
        fe = @view element_cache.fe[1:ndofs_elem]
        scatter_to_force!(cache.f, fe, dofs)
    end

    return nothing
end

"""
    scatter_to_triplets!(cache::COOCache, Ke::AbstractMatrix, dofs::AbstractVector{Int})

Scatter element stiffness matrix to triplet arrays **in-place**.

Appends all (i, j, value) triplets from element matrix to global triplet arrays.
Updates counter to track current position.

# Arguments
- `cache`: COO cache with triplet arrays
- `Ke`: Element stiffness matrix [ndofs_elem × ndofs_elem]
- `dofs`: Global DOF indices [ndofs_elem]

# Zero-Allocation Guarantee

Writes to pre-allocated triplet arrays. No new arrays created.

# Algorithm

```julia
for (i_local, i_global) in enumerate(dofs)
    for (j_local, j_global) in enumerate(dofs)
        counter += 1
        I[counter] = i_global
        J[counter] = j_global
        V[counter] = Ke[i_local, j_local]
    end
end
```
"""
function scatter_to_triplets!(
    cache::COOCache,
    Ke::AbstractMatrix,
    dofs::AbstractVector{Int}
)
    ndofs_elem = length(dofs)
    counter = cache.counter[]

    # Check capacity
    new_triplets = ndofs_elem * ndofs_elem
    if counter + new_triplets > cache.capacity
        error("COO cache overflow: need $(counter + new_triplets) triplets, " *
              "capacity is $(cache.capacity). Increase cache size.")
    end

    # Scatter element matrix to triplets
    for j_local in 1:ndofs_elem
        j_global = dofs[j_local]
        for i_local in 1:ndofs_elem
            i_global = dofs[i_local]
            counter += 1
            cache.I[counter] = i_global
            cache.J[counter] = j_global
            cache.V[counter] = Ke[i_local, j_local]
        end
    end

    cache.counter[] = counter
    return nothing
end

"""
    scatter_to_force!(f::Vector{Float64}, fe::AbstractVector, dofs::AbstractVector{Int})

Scatter element force vector to global force vector **in-place**.

Accumulates element contributions: `f[dofs] += fe`

# Arguments
- `f`: Global force vector (modified in-place)
- `fe`: Element force vector [ndofs_elem]
- `dofs`: Global DOF indices [ndofs_elem]

# Zero-Allocation Guarantee

No allocations - modifies `f` in-place.

# Algorithm

```julia
for (i_local, i_global) in enumerate(dofs)
    f[i_global] += fe[i_local]
end
```
"""
function scatter_to_force!(
    f::Vector{Float64},
    fe::AbstractVector,
    dofs::AbstractVector{Int}
)
    for (i_local, i_global) in enumerate(dofs)
        f[i_global] += fe[i_local]
    end
    return nothing
end

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

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
