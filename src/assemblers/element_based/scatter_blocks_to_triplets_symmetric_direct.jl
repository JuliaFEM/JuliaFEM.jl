# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
    scatter_blocks_to_triplets_symmetric_direct!(
        I::Vector{Int},
        J::Vector{Int},
        V::Vector{Float64},
        counter::Int,
        capacity::Int,
        K_blocks::Matrix{Tensor{2,3,Float64,9}},
        dofs::Vector{Int},
        N::Int
    ) -> Int

Direct version of scatter function that takes raw arrays instead of cache struct.

This version is designed to test if cache indirection causes dynamic dispatch.
Instead of passing `cache::COOCache`, we pass the raw arrays directly.

# Arguments
- `I`: Row indices array (modified in-place)
- `J`: Column indices array (modified in-place)
- `V`: Values array (modified in-place)
- `counter`: Current position in triplet arrays
- `capacity`: Maximum capacity of triplet arrays
- `K_blocks`: Element stiffness as [N×N] matrix of 3×3 tensor blocks (only upper triangle valid)
- `dofs`: Global DOF indices [3*N]
- `N`: Number of nodes in element

# Returns
- New counter position after adding triplets

# Zero-Allocation Guarantee

Writes directly to pre-allocated arrays. No intermediate structures created.

# Algorithm

Same as original scatter_blocks_to_triplets_symmetric! but without cache indirection:
```julia
for k in 1:N, l in k:N  # Upper triangle only
    block = K_blocks[k, l]
    if k == l
        # Diagonal block: add 9 triplets
    else
        # Off-diagonal block: add 18 triplets (both K[k,l] and K[l,k])
    end
end
return new_counter
```
"""
function scatter_blocks_to_triplets_symmetric_direct!(
    I::Vector{Int},
    J::Vector{Int},
    V::Vector{Float64},
    counter::Int,
    capacity::Int,
    K_blocks::Matrix{Tensor{2,3,Float64,9}},
    dofs::Vector{Int},
    N::Int
)::Int
    # Each diagonal block contributes 9 triplets
    # Each off-diagonal block contributes 18 triplets (9 for K[k,l] + 9 for K[l,k])
    n_diagonal = N
    n_offdiagonal = (N * (N - 1)) ÷ 2
    new_triplets = 9 * n_diagonal + 18 * n_offdiagonal

    if counter + new_triplets > capacity
        error("COO cache overflow: need $(counter + new_triplets) triplets, " *
              "capacity is $(capacity). Increase cache size.")
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
                I[counter] = i_global
                J[counter] = j_global
                V[counter] = block[α, β]
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
                I[counter] = i_global
                J[counter] = j_global
                V[counter] = value

                # K[l,k] contribution (symmetric): row from node l, col from node k
                counter += 1
                I[counter] = j_global  # Swap row and col
                J[counter] = i_global
                V[counter] = value     # Same value (symmetry)
            end
        end
    end

    return counter
end
