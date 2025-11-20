# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
    scatter_blocks_to_triplets!(
        cache::COOCache,
        K_blocks::AbstractMatrix{<:Tensor{2,3}},
        dofs::AbstractVector{Int},
        N::Int
    )

Scatter blocked tensor stiffness matrix directly to triplet arrays **in-place**.

Appends all (i, j, value) triplets from blocked tensor matrix to global triplet arrays.
This avoids the intermediate conversion to Float64 matrix.

# Arguments
- `cache`: COO cache with triplet arrays
- `K_blocks`: Element stiffness as [N×N] matrix of 3×3 tensor blocks
- `dofs`: Global DOF indices [3*N]
- `N`: Number of nodes in element

# Zero-Allocation Guarantee

Writes directly to pre-allocated triplet arrays. No intermediate matrices created.

# Algorithm

```julia
for k in 1:N, l in 1:N
    block = K_blocks[k, l]
    k_offset = 3(k - 1)
    l_offset = 3(l - 1)
    for α in 1:3, β in 1:3
        counter += 1
        i_global = dofs[k_offset + α]
        j_global = dofs[l_offset + β]
        I[counter] = i_global
        J[counter] = j_global
        V[counter] = block[α, β]
    end
end
```
"""
@inline function scatter_blocks_to_triplets!(
    cache::COOCache{EC,MC},
    K_blocks::Matrix{Tensor{2,3,Float64,9}},
    dofs::Vector{Int},
    N::Int
) where {EC,MC}
    counter = cache.counter[]
    ndofs_elem = 3 * N

    # Check capacity
    new_triplets = ndofs_elem * ndofs_elem
    if counter + new_triplets > cache.capacity
        error("COO cache overflow: need $(counter + new_triplets) triplets, " *
              "capacity is $(cache.capacity). Increase cache size.")
    end

    # Scatter blocks directly to triplets
    @inbounds for k in 1:N, l in 1:N
        block = K_blocks[k, l]
        k_offset = 3(k - 1)
        l_offset = 3(l - 1)
        for α in 1:3, β in 1:3
            counter += 1
            i_global = dofs[k_offset+α]
            j_global = dofs[l_offset+β]
            cache.I[counter] = i_global
            cache.J[counter] = j_global
            cache.V[counter] = block[α, β]
        end
    end

    cache.counter[] = counter
    return nothing
end
