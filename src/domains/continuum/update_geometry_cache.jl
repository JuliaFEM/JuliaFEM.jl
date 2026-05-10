# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

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
- `geometry_cache.∇N_data[ip, k]` — physical gradients `∇N` as `Vec{3}` (tangent to the
  embedded surface for `dim(topology)==2`; all three components may be nonzero)
- `geometry_cache.detJ_w[ip]` — `det(J) * w` for `D==3`, or `√(det G) · w` for `D==2`

The function is allocation-free; it reads topology, basis, and the
integration points from `element_cache` and writes back into the
pre-allocated `geometry_cache` arrays.

For **`dim(topology) == 3`**, the Jacobian `J = Σ_k X_k ⊗ ∇_ξ N_k` is a `Tensor{2,3}`,
inverted in the usual way, and `∇N` uses the full `Vec{3}` chain rule.

For **`dim(topology) == 2`**, node coordinates are `Vec{3}` but the isoparametric map
`x(ξ, η) ∈ ℝ³` is only two-parameter. Let `v_α = ∂x/∂ξ^α = Σ_k X_k ∂N_k/∂ξ^α` for
`α ∈ {1,2}` (columns of the `3 × 2` Jacobian), `G_{αβ} = v_α · v_β` (Gram matrix),
`detJ_w = √(det G) · w` (surface area measure on the embedded patch), and
`∇N_k = v_1 (G^{-1} ∂_ξ N_k)_1 + v_2 (G^{-1} ∂_ξ N_k)_2`. This reduces to the former
`(x, y)` / `J_2^{-T}` formula when the element lies in the global `xy` plane.
Degenerate `det(G) ≤ 0` throws `ArgumentError`.
"""
@inline function update_geometry_cache!(
    geometry_cache::GeometryCache,
    element_cache::ContinuumElementCache,
    elem_id::Int,
    mesh::AbstractMesh,
)
    conn = mesh.connectivity[elem_id]
    nnodes = length(conn)

    @inbounds for i in 1:nnodes
        node = conn[i]
        geometry_cache.X[i] = mesh.nodes[node]
    end

    ips = element_cache.ips
    nips = length(ips)
    D = dim(element_cache.topology)

    if D == 3
        @inbounds for ip_idx in 1:nips
            ip = ips[ip_idx]
            ξ = ip.coords

            N_vals = get_basis_functions(  element_cache.topology, element_cache.basis, ξ)
            dN_dξ  = get_basis_derivatives(element_cache.topology, element_cache.basis, ξ)

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
    elseif D == 2
        F = geometry_eltype(geometry_cache)
        @inbounds for ip_idx in 1:nips
            ip = ips[ip_idx]
            ξ = ip.coords

            N_vals = get_basis_functions(  element_cache.topology, element_cache.basis, ξ)
            dN_dξ  = get_basis_derivatives(element_cache.topology, element_cache.basis, ξ)

            v1 = zero(Vec{3,F})
            v2 = zero(Vec{3,F})
            for i in 1:nnodes
                xk = geometry_cache.X[i]
                di = dN_dξ[i]
                v1 += xk * di[1]
                v2 += xk * di[2]
            end
            G11 = v1 ⋅ v1
            G12 = v1 ⋅ v2
            G22 = v2 ⋅ v2
            G = Tensor{2,2,F,4}((G11, G12, G12, G22))
            detG = det(G)
            if !(detG > zero(F))
                throw(ArgumentError(
                    "update_geometry_cache!: singular or non-right-handed 2D element " *
                    "(det(G) = $detG); check node ordering and geometry",
                ))
            end
            Ginv = inv(G)
            detJ_w = sqrt(detG) * ip.weight

            for k in 1:nnodes
                geometry_cache.N_data[ip_idx, k] = N_vals[k]
                h = Ginv ⋅ dN_dξ[k]
                g = v1 * h[1] + v2 * h[2]
                geometry_cache.∇N_data[ip_idx, k] = Vec{3,F}((g[1], g[2], g[3]))
            end

            geometry_cache.detJ_w[ip_idx] = detJ_w
        end
    else
        throw(ArgumentError(
            "update_geometry_cache!: topology spatial dimension $D is not supported " *
            "(expected 2 or 3 for continuum assembly)",
        ))
    end

    return nothing
end
