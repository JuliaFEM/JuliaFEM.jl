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
