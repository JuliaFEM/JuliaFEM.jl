# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
    scatter_blocks_to_triplets_symmetric!(
        cache::COOCache,
        K_blocks::AbstractMatrix{<:Tensor{2,3}},
        dofs::AbstractVector{Int},
        N::Int
    )

Scatter blocked tensor stiffness matrix directly to triplet arrays **exploiting symmetry**.

Only upper triangle blocks (k ≤ l) are assumed to be computed. For each block:
- Diagonal blocks (k == l): Add once
- Off-diagonal blocks (k < l): Add both K[k,l] and K[l,k] (symmetric)

This halves the assembly cost and memory usage.

# Arguments
- `cache`: COO cache with triplet arrays
- `K_blocks`: Element stiffness as [N×N] matrix of 3×3 tensor blocks (only upper triangle valid)
- `dofs`: Global DOF indices [3*N]
- `N`: Number of nodes in element

# Zero-Allocation Guarantee

Writes directly to pre-allocated triplet arrays. No intermediate matrices created.

# Algorithm

```julia
for k in 1:N, l in k:N  # Upper triangle only
    block = K_blocks[k, l]
    # Add block[α,β] to (i,j) triplet
    # If k != l, also add block[β,α] to (j,i) triplet (symmetry)
end
```
"""
@inline function scatter_blocks_to_triplets_symmetric!(
    cache::COOCache{EC,MC},
    K_blocks::Matrix{Tensor{2,3,Float64,9}},
    dofs::Vector{Int},
    N::Int
) where {EC,MC}
    counter = cache.counter[]

    # Each diagonal block contributes 9 triplets
    # Each off-diagonal block contributes 18 triplets (9 for K[k,l] + 9 for K[l,k])
    n_diagonal = N
    n_offdiagonal = (N * (N - 1)) ÷ 2
    new_triplets = 9 * n_diagonal + 18 * n_offdiagonal

    if counter + new_triplets > cache.capacity
        error("COO cache overflow: need $(counter + new_triplets) triplets, " *
              "capacity is $(cache.capacity). Increase cache size.")
    end

    # Scatter upper triangle blocks
    @inbounds for k in 1:N, l in k:N
        block = K_blocks[k, l]
        k_offset = 3(k - 1)
        l_offset = 3(l - 1)

        if k == l
            # Diagonal block: add once
            for α in 1:3, β in 1:3
                counter += 1
                i_global = dofs[k_offset+α]
                j_global = dofs[l_offset+β]
                cache.I[counter] = i_global
                cache.J[counter] = j_global
                cache.V[counter] = block[α, β]
            end
        else
            # Off-diagonal block: add both K[k,l] and K[l,k]
            # For symmetry: K_global[i_k_α, j_l_β] == K_global[j_l_β, i_k_α]
            # So: K_blocks[k,l][α,β] contributes to both (i_k_α, j_l_β) and (j_l_β, i_k_α)
            for α in 1:3, β in 1:3
                value = block[α, β]

                # K[k,l] contribution: row from node k, col from node l
                counter += 1
                i_global = dofs[k_offset+α]
                j_global = dofs[l_offset+β]
                cache.I[counter] = i_global
                cache.J[counter] = j_global
                cache.V[counter] = value

                # K[l,k] contribution (symmetric): row from node l, col from node k
                counter += 1
                cache.I[counter] = j_global  # Swap row and col
                cache.J[counter] = i_global
                cache.V[counter] = value     # Same value (symmetry)
            end
        end
    end

    cache.counter[] = counter
    return nothing
end
