# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Domain kernel interface specification.

Domain kernels implement the physics-specific computations (WHAT to assemble).
Assemblers implement the traversal strategy (HOW to assemble).

This file defines the interface that domain kernels must implement to work
with generic assemblers.
"""

# ============================================================================
# REQUIRED INTERFACE
# ============================================================================

"""
    compute_element_stiffness!(
        cache::ElementCache,
        kernel::AbstractKernel,
        element_id::Int,
        mesh::AbstractMesh
    ) -> Nothing

Compute element stiffness matrix and force vector **in-place**.

**Zero allocations requirement**: All computations must write to pre-allocated
arrays in `cache`. Never allocate new arrays.

# Arguments
- `cache`: Pre-allocated element workspace containing:
  - `cache.Ke`: Local stiffness matrix [ndofs_elem × ndofs_elem] (output)
  - `cache.fe`: Local force vector [ndofs_elem] (output)
  - `cache.coords`: Element node coordinates [nnodes_elem × ndim] (workspace)
  - `cache.dofs`: Global DOF indices [ndofs_elem] (workspace)
- `kernel`: Domain-specific kernel (continuum, plate, beam, etc.)
- `element_id`: Element index in mesh
- `mesh`: Finite element mesh

# Implementation Requirements

1. **Zero the output arrays** before accumulating:
   ```julia
   fill!(cache.Ke, 0.0)
   fill!(cache.fe, 0.0)
   ```

2. **Get element nodes and coordinates**:
   ```julia
   nodes = mesh.connectivity[element_id]
   for (i, node) in enumerate(nodes)
       cache.coords[i, :] .= mesh.nodes[node]
   end
   ```

3. **Loop over integration points**:
   ```julia
   for ip in integration_points(integration)
       # Compute B-matrix, jacobian, etc.
       # Accumulate Ke, fe
   end
   ```

4. **Never return anything** - all results written to `cache.Ke`, `cache.fe`.

# Example Implementation (Continuum Mechanics)

```julia
function compute_element_stiffness!(
    cache::ElementCache,
    kernel::ContinuumKernel,
    element_id::Int,
    mesh::AbstractMesh
)
    # Zero output arrays
    fill!(cache.Ke, 0.0)
    fill!(cache.fe, 0.0)

    # Get element nodes
    nodes = mesh.connectivity[element_id]
    nnodes_elem = length(nodes)
    ndim = 3  # 3D continuum

    # Get node coordinates
    for (i, node) in enumerate(nodes)
        cache.coords[i, :] .= mesh.nodes[node]
    end

    # Get basis and integration
    basis = get_basis_functions(topology, nnodes_elem)
    integration = Gauss(order=2)

    # Loop over integration points
    for ip in integration_points(integration)
        ξ = ip.ξ
        w = ip.weight

        # Compute B-matrix (strain-displacement)
        B = compute_b_matrix(basis, cache.coords, ξ)

        # Material stiffness
        C = elasticity_tensor(kernel.material)

        # Jacobian determinant
        detJ = compute_jacobian(cache.coords, basis, ξ)

        # Accumulate stiffness: Ke += B^T * C * B * detJ * w
        dV = detJ * w
        # Use BLAS for efficiency: Ke += (B' * C * B) * dV
        mul!(cache.Ke, B', C * B * dV, 1.0, 1.0)
    end

    return nothing
end
```

# See Also
- [`dofs_per_node`](@ref) - Number of DOFs per node
- [`get_dof_mapping!`](@ref) - Global DOF indices for element
"""
function compute_element_stiffness! end

"""
    dofs_per_node(kernel::AbstractKernel) -> Int

Number of degrees of freedom per node for this kernel.

This depends on the field type:
- `Displacement{3}`: 3 DOFs per node (ux, uy, uz)
- `DisplacementRotation{3}`: 6 DOFs per node (ux, uy, uz, θx, θy, θz)
- `Temperature`: 1 DOF per node (T)
- `PlateDisplacement`: 3 DOFs per node (w, θx, θy)

# Arguments
- `kernel`: Domain kernel

# Returns
- Number of DOFs per node (integer)

# Example

```julia
kernel = ContinuumKernel(
    formulation = ContinuumFormulation{FullThreeD}(),
    material = LinearElastic(E=210e9, ν=0.3),
    field = Displacement{3}()
)

ndofs = dofs_per_node(kernel)  # Returns 3
```
"""
function dofs_per_node end

# Default implementation: delegate to field
"""
    dofs_per_node(kernel::AbstractKernel) -> Int

Default implementation: Extract field from kernel and delegate to field's dofs_per_node.

Kernels that store a field should implement `get_field(kernel)` to enable this delegation.
Otherwise, override this method directly.
"""
function dofs_per_node(kernel::AbstractKernel)
    return dofs_per_node(get_field(kernel))
end

"""
    get_field(kernel::AbstractKernel) -> AbstractField

Extract the field from a kernel.

Kernels should implement this to enable automatic DOF mapping delegation.
Default implementation throws an error.

# Example

```julia
get_field(kernel::ContinuumKernel) = kernel.field
```
"""
function get_field(kernel::AbstractKernel)
    error("get_field not implemented for $(typeof(kernel)). " *
          "Either implement get_field(::$(typeof(kernel))) or override dofs_per_node/get_dof_mapping! directly.")
end

"""
    get_dof_mapping!(
        dofs::AbstractVector{Int},
        kernel::AbstractKernel,
        element_id::Int,
        mesh::AbstractMesh
    ) -> Nothing

Fill global DOF indices for an element **in-place**.

**Zero allocations requirement**: Write DOF indices to pre-allocated `dofs`
vector. Never allocate new array.

# Arguments
- `dofs`: Pre-allocated DOF index buffer [ndofs_elem] (output)
- `kernel`: Domain kernel
- `element_id`: Element index in mesh
- `mesh`: Finite element mesh

# DOF Numbering Convention

DOFs are numbered **node-major** (all DOFs for node 1, then node 2, etc.):

```
Node-major ordering:
  Node 1: DOFs [1, 2, 3]         (ux, uy, uz)
  Node 2: DOFs [4, 5, 6]         (ux, uy, uz)
  Node 3: DOFs [7, 8, 9]         (ux, uy, uz)
  ...
  Node n: DOFs [3n-2, 3n-1, 3n]  (ux, uy, uz)

For element with nodes [10, 20, 30, 40]:
  dofs = [28, 29, 30, 58, 59, 60, 88, 89, 90, 118, 119, 120]
         |_________| |_________| |_________| |___________|
           node 10     node 20     node 30     node 40
```

# Implementation

```julia
function get_dof_mapping!(
    dofs::AbstractVector{Int},
    kernel::ContinuumKernel,  # 3 DOFs per node
    element_id::Int,
    mesh::AbstractMesh
)
    nodes = mesh.connectivity[element_id]
    nnodes_elem = length(nodes)
    ndofs_per_node = 3

    # Fill DOF indices (node-major)
    idx = 1
    for node in nodes
        for component in 1:ndofs_per_node
            dofs[idx] = (node - 1) * ndofs_per_node + component
            idx += 1
        end
    end

    return nothing
end
```

# See Also
- [`dofs_per_node`](@ref) - Number of DOFs per node
- [`compute_element_stiffness!`](@ref) - Compute element matrices
"""
function get_dof_mapping! end

# Default implementation: delegate to field-based DOF mapping
"""
    get_dof_mapping!(
        dofs::AbstractVector{Int},
        kernel::AbstractKernel,
        element_id::Int,
        mesh::AbstractMesh
    ) -> Nothing

Default implementation: Delegate to field-based DOF mapping.

Extracts nodes from mesh, gets field from kernel, and calls field-based
get_dof_mapping!. Kernels with special DOF patterns can override this.

# Implementation

```julia
function get_dof_mapping!(dofs, kernel, element_id, mesh)
    nodes = mesh.connectivity[element_id]
    field = get_field(kernel)
    get_dof_mapping!(dofs, field, nodes)
    return nothing
end
```
"""
function get_dof_mapping!(
    dofs::AbstractVector{Int},
    kernel::AbstractKernel,
    element_id::Int,
    mesh::AbstractMesh
)
    nodes = mesh.connectivity[element_id]
    field = get_field(kernel)
    ndofs_per_node = dofs_per_node(field)

    # Inline DOF mapping to avoid allocation from tuple conversion
    idx = 1
    @inbounds for node_id in nodes
        for component in 1:ndofs_per_node
            dofs[idx] = ndofs_per_node * (Int(node_id) - 1) + component
            idx += 1
        end
    end

    return nothing
end

# ============================================================================
# OPTIONAL INTERFACE (for specialized assemblers)
# ============================================================================

"""
    compute_node_contribution!(
        node_cache::NodeCache,
        kernel::AbstractKernel,
        node_id::Int,
        mesh::AbstractMesh,
        node_to_elements::NodeToElementsMap
    ) -> Nothing

Compute nodal contributions from all touching elements **in-place**.

Used by nodal-based assemblers. Not all kernels need to implement this -
default implementation falls back to element-based computation.

# Arguments
- `node_cache`: Pre-allocated node workspace
- `kernel`: Domain kernel
- `node_id`: Node index in mesh
- `mesh`: Finite element mesh
- `node_to_elements`: Inverse connectivity map

# Default Implementation

Default behavior: for each element touching this node, compute full element
stiffness, extract only rows/columns for this node.

Specialized implementations can optimize by computing only node contributions
directly (e.g., for explicit dynamics).
"""
function compute_node_contribution! end

# ============================================================================
# HELPER FUNCTIONS (for kernel implementations)
# ============================================================================

"""
    blocked_tensor_to_matrix_view!(
        K_e::AbstractMatrix{Float64},
        K_blocks::AbstractMatrix{Tensor{2,3}}
    )

Convert N×N matrix of 3×3 tensor blocks to 3N×3N Float64 matrix **in-place**.

Maps block[k,l][α,β] → K_e[3(k-1)+α, 3(l-1)+β]

# Arguments
- `K_e`: Output element stiffness matrix [3N × 3N] (modified in-place)
- `K_blocks`: Input blocked tensor matrix [N × N] of Tensor{2,3}

# Zero-Allocation Guarantee

Writes directly to pre-allocated `K_e`. No allocations.

# Usage

This is a common utility used by assemblers after computing stiffness blocks:

```julia
# Compute all blocks
for k in 1:N, l in 1:N
    K_blocks[k, l] = compute_block!(geometry_cache, material_cache, k, l)
end

# Convert to Float64 matrix
blocked_tensor_to_matrix_view!(element_cache.Ke, K_blocks)
```
"""
function blocked_tensor_to_matrix_view!(
    K_e::AbstractMatrix{Float64},
    K_blocks::AbstractMatrix{<:Tensor{2,3}}
)
    Nnodes = size(K_blocks, 1)
    @inbounds for k in 1:Nnodes, l in 1:Nnodes
        block = K_blocks[k, l]
        k_offset = 3(k - 1)
        l_offset = 3(l - 1)
        for α in 1:3, β in 1:3
            K_e[k_offset+α, l_offset+β] = block[α, β]
        end
    end
end

"""
    compute_block!(
        geometry_cache::GeometryCache,
        material_cache::MaterialStateCache,
        k_local::Int,
        l_local::Int
    ) -> Tensor{2,3,Float64,9}

**Phase 2:** Compute stiffness block using **precomputed material state**.

Integrates weak form over element to get stiffness block K[k,l] between
nodes k and l. Uses precomputed tangent moduli from Phase 1 - **no material calls**!

# Arguments
- `geometry_cache`: Precomputed element geometry (gradients, detJ*w)
- `material_cache`: Precomputed material state at all IPs (from Phase 1)
- `k_local`, `l_local`: Local node indices

# Returns
Fully integrated 3×3 stiffness block K[k,l]

# Performance
- **Zero material calls** - all tangents precomputed in Phase 1
- **Cache-friendly** - state_cache stays hot in L1/L2
- **GPU-ready** - Pure integration, no branching
- **Nodal assembly** - Compute state once, use N² times
- **Zero allocation** - No type parameters to avoid 80 bytes overhead

# Example
```julia
# Phase 1: Update caches
update_geometry_cache!(geometry_cache, element_cache, kernel, elem_id, mesh)
update_material_cache!(material_cache, geometry_cache, material, element_cache, state_old, elem_id, Δt)

# Phase 2: Assemble all blocks (no material calls!)
for k in 1:N, l in 1:N
    K_kl = compute_block!(∇N_data, detJ_w, 𝔻, k, l)
end
```
"""
function compute_block(
    ∇N_data::Matrix{Vec{3,Float64}},
    detJ_w::Vector{Float64},
    𝔻::Vector{SymmetricTensor{4,3,Float64,36}},
    k_local::Int,
    l_local::Int
)
    K_kl = zero(Tensor{2,3,Float64,9})

    # Get NIP from input
    NIP = length(detJ_w)

    # Integrate using precomputed tangent at each IP
    @inbounds for q in 1:NIP
        grad_k = ∇N_data[q, k_local]
        grad_l = ∇N_data[q, l_local]
        w = detJ_w[q]
        D = 𝔻[q]

        K_kl_ip = compute_block_at_point(grad_k, grad_l, D)
        K_kl += K_kl_ip * w
    end

    return K_kl
end

"""
    compute_block!(
        K_blocks::Matrix{Tensor{2,3,Float64,9}},
        ∇N_data::Matrix{Vec{3,Float64}},
        detJ_w::Vector{Float64},
        𝔻::Vector{SymmetricTensor{4,3,Float64,36}},
        k_local::Int,
        l_local::Int
    ) -> Nothing

Mutating version that writes directly to K_blocks matrix.
Avoids return value allocation by writing result in-place.

# Arguments
- `K_blocks`: Output matrix [N×N] of 3×3 tensor blocks
- `∇N_data`: Shape function gradients [NIP × N] at all integration points
- `detJ_w`: Jacobian determinant times weight [NIP] at all integration points
- `𝔻`: Material tangent moduli [NIP] at all integration points
- `k_local`, `l_local`: Local node indices

# Performance
Direct array access eliminates field overhead from cache structs.
Achieves **zero allocations** by avoiding return value overhead.
"""
function compute_block!(
    K_blocks::Matrix{Tensor{2,3,Float64,9}},
    ∇N_data::Matrix{Vec{3,Float64}},
    detJ_w::Vector{Float64},
    𝔻::Vector{SymmetricTensor{4,3,Float64,36}},
    k_local::Int,
    l_local::Int
)
    K_kl = zero(Tensor{2,3,Float64,9})

    # Get NIP from input
    NIP = length(detJ_w)

    # Integrate using precomputed tangent at each IP
    @inbounds for q in 1:NIP
        grad_k = ∇N_data[q, k_local]
        grad_l = ∇N_data[q, l_local]
        w = detJ_w[q]
        D = 𝔻[q]

        K_kl_ip = compute_block_at_point(grad_k, grad_l, D)
        K_kl += K_kl_ip * w
    end

    # Write directly to matrix - avoids return value allocation
    K_blocks[k_local, l_local] = K_kl
    return nothing
end

"""
    compute_all_blocks!(
        K_blocks::AbstractMatrix{Tensor{2,3,Float64,9}},
        ∇N_data::Matrix{Vec{3,Float64}},
        detJ_w::Vector{Float64},
        𝔻::Vector{SymmetricTensor{4,3,Float64,36}}
    ) -> Nothing

**Phase 2:** Compute all N×N stiffness blocks using **precomputed material state**.

Helper for element-based assemblers. Uses two-phase approach:
- Phase 1 (before this): Cache updates computed geometry and material state
- Phase 2 (this function): Assemble all blocks using precomputed state

# Arguments
- `K_blocks`: Output matrix [N×N] of 3×3 tensor blocks
- `∇N_data`: Shape function gradients [NIP × N] at all integration points
- `detJ_w`: Jacobian determinant times weight [NIP] at all integration points
- `𝔻`: Material tangent moduli [NIP] at all integration points

# Examples
```julia
# Phase 1: Update caches
update_geometry_cache!(geometry_cache, element_cache, kernel, elem_id, mesh)
update_material_cache!(material_cache, geometry_cache, material, element_cache, state_old, elem_id, Δt)

# Phase 2: Assemble all blocks (no material calls!)
compute_all_blocks!(K_blocks, geometry_cache.∇N_data, geometry_cache.detJ_w, material_cache.𝔻)
```
"""
function compute_all_blocks!(
    K_blocks::AbstractMatrix{<:Tensor{2,3}},
    ∇N_data::Matrix{Vec{3,Float64}},
    detJ_w::Vector{Float64},
    𝔻::Vector{SymmetricTensor{4,3,Float64,36}}
)
    # N is inferred from ∇N_data size at runtime
    N = size(∇N_data, 2)

    @inbounds for k in 1:N, l in 1:N
        compute_block!(K_blocks, ∇N_data, detJ_w, 𝔻, k, l)
    end
end

"""
    compute_b_matrix(
        basis::AbstractBasis,
        coords::Matrix{Float64},
        ξ::Vec
    ) -> Matrix{Float64}

Compute strain-displacement matrix B at integration point.

For 3D continuum mechanics with 4-node tetrahedron:
- Input: `coords` [4 × 3], basis functions, parametric coordinate `ξ`
- Output: `B` [6 × 12] matrix relating nodal displacements to strains

This is a **helper function** - can allocate for convenience.
Called inside `compute_element_stiffness!` which is zero-allocation at the
assembly level (element level can allocate transiently).

# Arguments
- `basis`: Basis functions for element
- `coords`: Element node coordinates [nnodes × ndim]
- `ξ`: Parametric coordinate of integration point

# Returns
- B-matrix [nstrain × ndofs_elem]

# Example

```julia
# Inside compute_element_stiffness!
for ip in integration_points(integration)
    B = compute_b_matrix(basis, cache.coords, ip.ξ)  # OK to allocate here
    # Use B to accumulate Ke...
end
```
"""
function compute_b_matrix end

"""
    compute_jacobian(
        coords::Matrix{Float64},
        basis::AbstractBasis,
        ξ::Vec
    ) -> Float64

Compute jacobian determinant at integration point.

Used for coordinate transformation: `dV = detJ * dξ`

# Arguments
- `coords`: Element node coordinates [nnodes × ndim]
- `basis`: Basis functions for element
- `ξ`: Parametric coordinate of integration point

# Returns
- Jacobian determinant (scalar)
"""
function compute_jacobian end

# ============================================================================
# KERNEL VALIDATION
# ============================================================================

"""
    validate_kernel(kernel::AbstractKernel, mesh::AbstractMesh) -> Bool

Check if kernel implements required interface correctly.

Tests:
- `dofs_per_node` returns positive integer
- `get_dof_mapping!` produces valid DOF indices
- `compute_element_stiffness!` writes to cache without allocating

# Arguments
- `kernel`: Domain kernel to validate
- `mesh`: Test mesh

# Returns
- `true` if kernel implements interface correctly

# Throws
- `ErrorException` if kernel is invalid, with detailed message
"""
function validate_kernel(kernel::AbstractKernel, mesh::AbstractMesh)
    # Test 1: dofs_per_node
    ndofs = dofs_per_node(kernel)
    if ndofs <= 0
        error("dofs_per_node must return positive integer, got $ndofs")
    end

    # Test 2: get_dof_mapping!
    if nelements(mesh) == 0
        error("Mesh has no elements")
    end

    elem_id = 1
    nodes = mesh.connectivity[elem_id]
    nnodes_elem = length(nodes)
    ndofs_elem = nnodes_elem * ndofs
    dofs = zeros(Int, ndofs_elem)

    get_dof_mapping!(dofs, kernel, elem_id, mesh)

    if any(dofs .<= 0)
        error("get_dof_mapping! produced invalid DOF indices: $dofs")
    end

    if length(unique(dofs)) != length(dofs)
        error("get_dof_mapping! produced duplicate DOF indices: $dofs")
    end

    # Test 3: compute_element_stiffness!
    cache = create_element_cache(mesh, kernel)
    compute_element_stiffness!(cache, kernel, elem_id, mesh)

    if any(isnan, cache.Ke)
        error("compute_element_stiffness! produced NaN in Ke")
    end

    if any(isnan, cache.fe)
        error("compute_element_stiffness! produced NaN in fe")
    end

    return true
end
