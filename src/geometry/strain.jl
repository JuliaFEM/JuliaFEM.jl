# This file is part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE

"""
Strain computation utilities for finite element analysis.

This module provides functions for computing strain tensors from nodal
displacements and basis function derivatives.
"""

using Tensors

"""
    compute_strain(u_elem, dN_dx) -> SymmetricTensor{2,3}

Compute small strain tensor from nodal displacements.

Small strain: ε = ½(∇u + ∇u^T) where ∇u = ∑ᵢ uᵢ ⊗ (dNᵢ/dx)

# Arguments
- `u_elem::NTuple{N, Vec{3}}`: Nodal displacement vectors
- `dN_dx::NTuple{N, Vec{3}}`: Basis derivatives in physical coordinates

# Returns
- `ε::SymmetricTensor{2,3}`: Small strain tensor

# Example
```julia
u = (Vec{3}((0.1, 0.0, 0.0)), Vec{3}((0.15, 0.02, 0.0)))
dN_dx = (Vec{3}((-1.0, -1.0, -1.0)), Vec{3}((1.0, 0.0, 0.0)))
ε = compute_strain(u, dN_dx)
```
"""
function compute_strain(
    u_elem::NTuple{N,Vec{3}},
    dN_dx::NTuple{N,Vec{3}}
) where N
    # Displacement gradient: ∇u = ∑ᵢ uᵢ ⊗ (dNᵢ/dx)
    ∇u = sum(u_i ⊗ dN_i for (u_i, dN_i) in zip(u_elem, dN_dx))

    # Small strain (symmetric part): ε = ½(∇u + ∇u^T)
    ε = symmetric(∇u)

    return ε
end
