# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE

"""
    compute_jacobian(X, dN_dξ) -> Tensor{2, D}

Compute the Jacobian matrix J = ∂x/∂ξ at an integration point.

The Jacobian transforms derivatives from reference coordinates (ξ) to physical 
coordinates (x) via the isoparametric mapping.

# Arguments
- `X`: Element node coordinates in physical space (tuple or vector of `Vec{D}`)
- `dN_dξ`: Shape function derivatives in reference coordinates (tuple or vector of `Vec{D}`)

# Returns
- `J::Tensor{2, D}`: Jacobian matrix where `J[i,j] = ∂xᵢ/∂ξⱼ`

# Mathematical Definition
```
J = ∑ᵢ (dNᵢ/dξ) ⊗ Xᵢ
```

Where ⊗ is the tensor product (outer product).

# Example: Triangle P1 (2D)
```julia
using JuliaFEM
using Tensors

# Element nodes in physical space
X = (Vec{2}(0.0, 0.0), Vec{2}(2.0, 0.0), Vec{2}(0.0, 1.5))

# Get basis derivatives at integration point
xi = Vec{2}(1/3, 1/3)  # Center of reference triangle
dN_dξ = get_basis_derivatives(Triangle(), Lagrange{Triangle, 1}(), xi)
# Returns: (Vec(-1.0, -1.0), Vec(1.0, 0.0), Vec(0.0, 1.0))

# Compute Jacobian
J = compute_jacobian(X, dN_dξ)
# J = [2.0  0.0]
#     [0.0  1.5]

# Jacobian determinant (element area/volume scaling)
detJ = det(J)  # 3.0 (twice the triangle area)
```

# Example: Tetrahedron P1 (3D)
```julia
# Element nodes in physical space
X = (
    Vec{3}(0.0, 0.0, 0.0),
    Vec{3}(1.0, 0.0, 0.0),
    Vec{3}(0.0, 2.0, 0.0),
    Vec{3}(0.0, 0.0, 3.0)
)

# Get basis derivatives
xi = Vec{3}(0.25, 0.25, 0.25)  # Inside tetrahedron
dN_dξ = get_basis_derivatives(Tetrahedron(), Lagrange{Tetrahedron, 1}(), xi)

# Compute Jacobian
J = compute_jacobian(X, dN_dξ)
# J = [1.0  0.0  0.0]
#     [0.0  2.0  0.0]
#     [0.0  0.0  3.0]

detJ = det(J)  # 6.0
```

# Zero Allocation
This function is fully type-stable and zero-allocation when `X` and `dN_dξ` are 
tuples or `StaticVector`s of `Vec` types from Tensors.jl.

# See Also
- [`physical_derivatives`](@ref): Transform derivatives to physical coordinates
- [`get_basis_derivatives`](@ref): Compute shape function derivatives
- [`Tensor`](@ref): Tensors.jl tensor type
"""
function compute_jacobian(X::NTuple{N,Vec{D}}, dN_dξ::NTuple{N,Vec{D}}) where {N,D}
    # J = ∑ᵢ (dNᵢ/dξ) ⊗ Xᵢ
    # One-liner with Tensors.jl!
    return sum(dN_dξi ⊗ Xi for (dN_dξi, Xi) in zip(dN_dξ, X))
end

# Overload for AbstractVector inputs (less efficient, allocates)
function compute_jacobian(X::AbstractVector{<:Vec{D}}, dN_dξ::AbstractVector{<:Vec{D}}) where {D}
    N = length(X)
    @assert length(dN_dξ) == N "X and dN_dξ must have same length"
    return sum(dN_dξ[i] ⊗ X[i] for i in 1:N)
end

"""
    physical_derivatives(J, dN_dξ) -> Tuple of Vec{D}

Transform shape function derivatives from reference to physical coordinates.

# Mathematical Definition
```
dNᵢ/dx = (J⁻¹)ᵀ ⋅ (dNᵢ/dξ)
```

# Arguments
- `J::Tensor{2, D}`: Jacobian matrix from `compute_jacobian`
- `dN_dξ`: Shape function derivatives in reference coordinates (tuple of `Vec{D}`)

# Returns
- Tuple of `Vec{D}`: Shape function derivatives in physical coordinates

# Example
```julia
using JuliaFEM
using Tensors

# Setup (from previous example)
X = (Vec{2}(0.0, 0.0), Vec{2}(2.0, 0.0), Vec{2}(0.0, 1.5))
xi = Vec{2}(1/3, 1/3)
dN_dξ = get_basis_derivatives(Triangle(), Lagrange{Triangle, 1}(), xi)

# Compute Jacobian
J = compute_jacobian(X, dN_dξ)

# Transform derivatives to physical coordinates
dN_dx = physical_derivatives(J, dN_dξ)
# dN_dx[1] = Vec(-0.5, -0.666...)  # ∂N₁/∂x, ∂N₁/∂y
# dN_dx[2] = Vec(0.5, 0.0)         # ∂N₂/∂x, ∂N₂/∂y
# dN_dx[3] = Vec(0.0, 0.666...)    # ∂N₃/∂x, ∂N₃/∂y

# Verification: ∑ᵢ dNᵢ/dx = 0 (constant strain condition)
sum(dN_dx)  # ≈ Vec(0.0, 0.0)
```

# Usage in Assembly
```julia
for ip in integration_points(Gauss{2}(), Triangle())
    xi = Vec(ip.ξ)
    
    # Basis evaluation
    N = get_basis_functions(Triangle(), Lagrange{Triangle, 1}(), xi)
    dN_dξ = get_basis_derivatives(Triangle(), Lagrange{Triangle, 1}(), xi)
    
    # Jacobian transformation
    J = compute_jacobian(X, dN_dξ)
    detJ = det(J)
    dN_dx = physical_derivatives(J, dN_dξ)
    
    # Use dN_dx for strain computation, stiffness assembly, etc.
    ε = compute_strain(u_elem, dN_dx)
    # ...
end
```

# Zero Allocation
Fully type-stable and zero-allocation when inputs are tuples of `Vec` types.

# See Also
- [`compute_jacobian`](@ref): Compute the Jacobian matrix
- [`compute_strain`](@ref): Compute strain from displacements
"""
function physical_derivatives(J::Tensor{2,D}, dN_dξ::NTuple{N,Vec{D}}) where {N,D}
    invJ_t = inv(J)'  # Transpose of inverse Jacobian
    return map(dN_i -> invJ_t ⋅ dN_i, dN_dξ)
end

# Overload for AbstractVector input
function physical_derivatives(J::Tensor{2,D}, dN_dξ::AbstractVector{<:Vec{D}}) where {D}
    invJ_t = inv(J)'
    return [invJ_t ⋅ dN_i for dN_i in dN_dξ]
end
