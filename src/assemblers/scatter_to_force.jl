# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
    scatter_to_force!(f::Vector{Float64}, fe::AbstractVector, dofs::AbstractVector{Int})

Scatter element force vector to global force vector **in-place** (legacy format).

Accumulates element contributions: `f[dofs] += fe`

# Arguments
- `f`: Global force vector (modified in-place)
- `fe`: Element force vector [ndofs_elem]
- `dofs`: Global DOF indices [ndofs_elem]

# Zero-Allocation Guarantee

No allocations - modifies `f` in-place.

# Algorithm

```julia
for (i_local, i_global) in enumerate(dofs)
    f[i_global] += fe[i_local]
end
```
"""
function scatter_to_force!(
    f::Vector{Float64},
    fe::Vector{Float64},
    dofs::Vector{Int}
)
    for (i_local, i_global) in enumerate(dofs)
        f[i_global] += fe[i_local]
    end
    return nothing
end
