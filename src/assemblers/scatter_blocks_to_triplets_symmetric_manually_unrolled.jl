# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
    scatter_blocks_to_triplets_symmetric_manually_unrolled!(
        cache::COOCache,
        K_blocks::Matrix{Tensor{2,3,Float64,9}},
        dofs::Vector{Int},
        N::Int
    )

Scatter blocked tensor stiffness matrix directly to triplet arrays **with manual loop unrolling**.

This version manually unrolls the inner 3×3 loops to eliminate dynamic dispatch on loop variables.
The compiler can then fully specialize all indexing operations at compile time.

Only upper triangle blocks (k ≤ l) are assumed to be computed. For each block:
- Diagonal blocks (k == l): Add once
- Off-diagonal blocks (k < l): Add both K[k,l] and K[l,k] (symmetric)

# Arguments
- `cache`: COO cache with triplet arrays
- `K_blocks`: Element stiffness as [N×N] matrix of 3×3 tensor blocks (only upper triangle valid)
- `dofs`: Global DOF indices [3*N]
- `N`: Number of nodes in element

# Zero-Allocation Guarantee

Writes directly to pre-allocated triplet arrays. No intermediate matrices created.
Manual unrolling eliminates dynamic dispatch on loop variables.

# Performance

This manually unrolled version eliminates the 18 dynamic dispatch sites found in
the original looped version, achieving true zero-allocation assembly.
"""
@inline function scatter_blocks_to_triplets_symmetric_manually_unrolled!(
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
            # Diagonal block: manually unrolled 3×3
            # Precompute all DOF indices
            i1 = dofs[k_offset+1]
            i2 = dofs[k_offset+2]
            i3 = dofs[k_offset+3]
            j1 = dofs[l_offset+1]
            j2 = dofs[l_offset+2]
            j3 = dofs[l_offset+3]
            
            # Row 1 (α=1)
            counter += 1; cache.I[counter] = i1; cache.J[counter] = j1; cache.V[counter] = block[1,1]
            counter += 1; cache.I[counter] = i1; cache.J[counter] = j2; cache.V[counter] = block[1,2]
            counter += 1; cache.I[counter] = i1; cache.J[counter] = j3; cache.V[counter] = block[1,3]
            
            # Row 2 (α=2)
            counter += 1; cache.I[counter] = i2; cache.J[counter] = j1; cache.V[counter] = block[2,1]
            counter += 1; cache.I[counter] = i2; cache.J[counter] = j2; cache.V[counter] = block[2,2]
            counter += 1; cache.I[counter] = i2; cache.J[counter] = j3; cache.V[counter] = block[2,3]
            
            # Row 3 (α=3)
            counter += 1; cache.I[counter] = i3; cache.J[counter] = j1; cache.V[counter] = block[3,1]
            counter += 1; cache.I[counter] = i3; cache.J[counter] = j2; cache.V[counter] = block[3,2]
            counter += 1; cache.I[counter] = i3; cache.J[counter] = j3; cache.V[counter] = block[3,3]
        else
            # Off-diagonal block: manually unrolled with symmetry
            # Precompute all DOF indices
            i1 = dofs[k_offset+1]
            i2 = dofs[k_offset+2]
            i3 = dofs[k_offset+3]
            j1 = dofs[l_offset+1]
            j2 = dofs[l_offset+2]
            j3 = dofs[l_offset+3]
            
            # Row 1 (α=1): K[k,l] and K[l,k] contributions
            v11 = block[1,1]; counter += 1; cache.I[counter] = i1; cache.J[counter] = j1; cache.V[counter] = v11
                              counter += 1; cache.I[counter] = j1; cache.J[counter] = i1; cache.V[counter] = v11
            v12 = block[1,2]; counter += 1; cache.I[counter] = i1; cache.J[counter] = j2; cache.V[counter] = v12
                              counter += 1; cache.I[counter] = j2; cache.J[counter] = i1; cache.V[counter] = v12
            v13 = block[1,3]; counter += 1; cache.I[counter] = i1; cache.J[counter] = j3; cache.V[counter] = v13
                              counter += 1; cache.I[counter] = j3; cache.J[counter] = i1; cache.V[counter] = v13
            
            # Row 2 (α=2): K[k,l] and K[l,k] contributions
            v21 = block[2,1]; counter += 1; cache.I[counter] = i2; cache.J[counter] = j1; cache.V[counter] = v21
                              counter += 1; cache.I[counter] = j1; cache.J[counter] = i2; cache.V[counter] = v21
            v22 = block[2,2]; counter += 1; cache.I[counter] = i2; cache.J[counter] = j2; cache.V[counter] = v22
                              counter += 1; cache.I[counter] = j2; cache.J[counter] = i2; cache.V[counter] = v22
            v23 = block[2,3]; counter += 1; cache.I[counter] = i2; cache.J[counter] = j3; cache.V[counter] = v23
                              counter += 1; cache.I[counter] = j3; cache.J[counter] = i2; cache.V[counter] = v23
            
            # Row 3 (α=3): K[k,l] and K[l,k] contributions
            v31 = block[3,1]; counter += 1; cache.I[counter] = i3; cache.J[counter] = j1; cache.V[counter] = v31
                              counter += 1; cache.I[counter] = j1; cache.J[counter] = i3; cache.V[counter] = v31
            v32 = block[3,2]; counter += 1; cache.I[counter] = i3; cache.J[counter] = j2; cache.V[counter] = v32
                              counter += 1; cache.I[counter] = j2; cache.J[counter] = i3; cache.V[counter] = v32
            v33 = block[3,3]; counter += 1; cache.I[counter] = i3; cache.J[counter] = j3; cache.V[counter] = v33
                              counter += 1; cache.I[counter] = j3; cache.J[counter] = i3; cache.V[counter] = v33
        end
    end

    cache.counter[] = counter
    return nothing
end
