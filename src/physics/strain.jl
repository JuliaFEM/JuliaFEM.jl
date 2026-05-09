# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Strain extraction from displacement gradients.

Strain is a physics concept derived from displacement fields in continuum mechanics.
"""

using Tensors

"""
    extract_strain(∇u::Tensor{2,3})

Extract strain tensor from displacement gradient.

# Formula
```
ε = sym(∇u)  # Symmetric part of displacement gradient
```

# Arguments
- `∇u`: Displacement gradient tensor

# Returns
- `ε`: Strain tensor (symmetric)

See also [`compute_strain`](@ref), which first assembles `∇u` from nodal
values and physical basis derivatives.

# Examples
```julia
∇u = Tensor{2,3}((1.0, 0.1, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0, 1.0))
ε = extract_strain(∇u)  # SymmetricTensor{2,3}
```
"""
function extract_strain(∇u::Tensor{2,3,T}) where T
    return symmetric(∇u)
end

"""
    extract_strain_rate(∇u̇::Tensor{2,3})

Extract strain rate from velocity gradient.

# Formula
```
ε̇ = sym(∇u̇)  # Symmetric part of velocity gradient
```

# Arguments
- `∇u̇`: Velocity gradient tensor (or displacement gradient rate)

# Returns
- `ε̇`: Strain rate tensor (symmetric)

# Examples
```julia
∇u̇ = Tensor{2,3}((0.01, 0.0, 0.0, 0.0, -0.01, 0.0, 0.0, 0.0, 0.0))
ε̇ = extract_strain_rate(∇u̇)  # SymmetricTensor{2,3}
```

# Note
For quasi-static analysis, compute from increments:
```julia
∇u_rate = (∇u_new - ∇u_old) / Δt
ε̇ = extract_strain_rate(∇u_rate)
```
"""
function extract_strain_rate(∇u̇::Tensor{2,3,T}) where T
    return symmetric(∇u̇)
end
