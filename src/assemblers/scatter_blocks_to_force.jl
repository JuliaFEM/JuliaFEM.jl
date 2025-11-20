# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
    scatter_blocks_to_force!(
        f::Vector{Float64},
        f_blocks::AbstractVector{Vec{3,Float64}},
        dofs::AbstractVector{Int},
        N::Int
    )

Scatter blocked force vector to global force vector **in-place**.

Accumulates element contributions from blocked structure.

# Arguments
- `f`: Global force vector (modified in-place)
- `f_blocks`: Element force as [N] vector of Vec{3} blocks
- `dofs`: Global DOF indices [3*N]
- `N`: Number of nodes in element

# Zero-Allocation Guarantee

No allocations - modifies `f` in-place.

# Algorithm

```julia
for k in 1:N
    block = f_blocks[k]
    k_offset = 3(k - 1)
    for α in 1:3
        i_global = dofs[k_offset + α]
        f[i_global] += block[α]
    end
end
```
"""
@inline function scatter_blocks_to_force!(
    f::Vector{Float64},
    f_blocks::Vector{Vec{3,Float64}},
    dofs::Vector{Int},
    N::Int
)
    @inbounds for k in 1:N
        block = f_blocks[k]
        k_offset = 3(k - 1)
        for α in 1:3
            i_global = dofs[k_offset+α]
            f[i_global] += block[α]
        end
    end
    return nothing
end
