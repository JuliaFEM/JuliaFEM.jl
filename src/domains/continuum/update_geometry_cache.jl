# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Geometry cache update functions for continuum elements.

Extracts node coordinates and computes physical gradients and Jacobian data.
"""

using Tensors

"""
    update_geometry_cache!(geometry_cache, element_cache, elem_id, mesh) -> Nothing

Update `geometry_cache` for the element `elem_id` of `mesh`. The cache
fields written are:

- `geometry_cache.X` — node coordinates (one entry per element node)
- `geometry_cache.N_data[ip, k]` — basis values
- `geometry_cache.∇N_data[ip, k]` — physical gradients `∇N`
- `geometry_cache.detJ_w[ip]` — `det(J) * w` for integration

The function is allocation-free; it reads topology, basis, and the
integration points from `element_cache` and writes back into the
pre-allocated `geometry_cache` arrays. For each integration point the
Jacobian `J = X ⊗ ∇_ξ N` is built on the fly, then physical gradients
are obtained via `J^{-T} · ∇_ξ N`.
"""
@inline function update_geometry_cache!(
    geometry_cache::GeometryCache,
    element_cache::ContinuumElementCache,
    elem_id::Int,
    mesh::AbstractMesh,
)
    conn = mesh.connectivity[elem_id]
    nnodes = length(conn)

    # Extract node coordinates (mesh.nodes already contains Vec{3}).
    # Indexed loop avoids the iterator allocation that `enumerate` introduces.
    @inbounds for i in 1:nnodes
        node = conn[i]
        geometry_cache.X[i] = mesh.nodes[node]
    end

    ips = element_cache.ips
    nips = length(ips)

    @inbounds for ip_idx in 1:nips
        ip = ips[ip_idx]
        ξ = ip.coords

        N_vals = get_basis_functions(  element_cache.topology, element_cache.basis, ξ)
        dN_dξ  = get_basis_derivatives(element_cache.topology, element_cache.basis, ξ)

        # Jacobian J = X ⊗ ∇_ξ N
        J = geometry_cache.X[1] ⊗ dN_dξ[1]
        for i in 2:nnodes
            J += geometry_cache.X[i] ⊗ dN_dξ[i]
        end

        J_inv_T = transpose(inv(J))

        for k in 1:nnodes
            geometry_cache.N_data[ip_idx, k]  = N_vals[k]
            geometry_cache.∇N_data[ip_idx, k] = J_inv_T ⋅ dN_dξ[k]
        end

        geometry_cache.detJ_w[ip_idx] = det(J) * ip.weight
    end

    return nothing
end
