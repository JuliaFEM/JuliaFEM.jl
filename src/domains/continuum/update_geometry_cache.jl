# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Geometry cache update functions for continuum elements.

Extracts node coordinates and computes physical gradients and Jacobian data.
"""

using Tensors

"""
    update_geometry_cache!(
        geometry_cache::GeometryCache,
        element_cache::ElementCache,
        kernel::AbstractKernel,
        elem_id::Int,
        mesh::AbstractMesh
    )

Update geometry cache for current element.

Computes:
- Node coordinates → geometry_cache.X
- Physical gradients ∇N at each integration point → geometry_cache.∇N_data
- Jacobian determinant × weight (detJ * w) at each IP → geometry_cache.detJ_w

# Arguments
- `geometry_cache`: Geometry cache to update
- `element_cache`: Element cache (provides topology, basis, integration points)
- `kernel`: Domain kernel
- `elem_id`: Current element ID
- `mesh`: Finite element mesh

# Side Effects
Mutates geometry_cache.X, geometry_cache.∇N_data, geometry_cache.detJ_w

# Zero-Allocation Guarantee
No allocations - writes to pre-allocated geometry_cache arrays.

# Implementation Notes
For each integration point:
1. Get reference gradients ∇_ξ N from basis
2. Compute Jacobian J = X ⊗ ∇_ξ N
3. Compute physical gradients ∇N = J^{-T} ⋅ ∇_ξ N
4. Store detJ * weight for integration
"""
function update_geometry_cache!(
    geometry_cache::GeometryCache,
    element_cache::ElementCache,
    kernel::AbstractKernel,
    elem_id::Int,
    mesh::AbstractMesh
)

    # FIXME: drop kernel argument if unused
    # FIXME: drop element_cache argument and explicitly pass topology, basis, ips

    # Get element connectivity
    conn = mesh.connectivity[elem_id]
    nnodes = length(conn)

    # Extract node coordinates (mesh.nodes already contains Vec{3})
    for (i, node) in enumerate(conn)
        geometry_cache.X[i] = mesh.nodes[node]
    end

    # Compute physical gradients and detJ*w at each integration point
    ips = element_cache.ips
    nips = length(ips)

    @inbounds for ip_idx in 1:nips
        ip = ips[ip_idx]
        ξ = ip.ξ

        # Reference gradients
        dN_dξ = get_basis_derivatives(element_cache.topology, element_cache.basis, ξ)

        # Jacobian: J = X ⊗ ∇_ξ N
        J = geometry_cache.X[1] ⊗ dN_dξ[1]
        for i in 2:nnodes
            J += geometry_cache.X[i] ⊗ dN_dξ[i]
        end

        J_inv_T = transpose(inv(J))

        # Physical gradients for all nodes: ∇N = J^{-T} ⋅ ∇_ξ N
        for k in 1:nnodes
            geometry_cache.∇N_data[ip_idx, k] = J_inv_T ⋅ dN_dξ[k]
        end

        # Store detJ * weight
        geometry_cache.detJ_w[ip_idx] = det(J) * ip.weight
    end

    return nothing
end
