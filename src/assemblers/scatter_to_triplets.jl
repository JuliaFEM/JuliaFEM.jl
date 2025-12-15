# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

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
    Ke::Matrix{Float64},
    dofs::Vector{Int}
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
