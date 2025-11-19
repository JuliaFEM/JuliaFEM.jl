# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
    deformation_gradient.jl

Zero-allocation computation of deformation gradient tensor F for finite element analysis.

This module implements the **nodal assembly** pattern from our golden standard:
`docs/src/book/multigpu_nodal_assembly.md`

# Mathematical Background

The deformation gradient F maps material coordinates X to spatial coordinates x:
    
    x = X + u(X)
    F = ∂x/∂X = I + ∂u/∂X = I + ∇u

For finite strain: F = I + ∇u (default)
For small strain:  F = I (displacement gradient ignored)

# Architecture

- Uses Tensors.jl for all tensor operations (Vec, Tensor)
- Zero allocations (all operations on stack)
- Type-stable (all types known at compile time)
- GPU-ready (immutable operations only)

# References

- Hughes, "The Finite Element Method", Section 6.2
- Bonet & Wood, "Nonlinear Continuum Mechanics", Chapter 3
- `docs/src/book/multigpu_nodal_assembly.md` (golden standard)
"""

using Tensors
using LinearAlgebra

"""
    StrainFormulation

Enum-like type for strain formulation selection.

# Values
- `FiniteStrain()`: Full nonlinear kinematics, F = I + ∇u
- `SmallStrain()`: Linearized kinematics, F = I (ignores displacement gradient)
"""
abstract type StrainFormulation end
struct FiniteStrain <: StrainFormulation end
struct SmallStrain <: StrainFormulation end

"""
    compute_deformation_gradient(
        X_nodes::NTuple{N, Vec{3, Float64}},
        u_nodes::NTuple{N, Vec{3, Float64}},
        dN_dξ::NTuple{N, Vec{D, Float64}},
        J::Tensor{2, 3, Float64, 9},
        formulation::StrainFormulation = FiniteStrain()
    ) -> Tensor{2, 3, Float64, 9}

Compute deformation gradient F at an integration point.

# Mathematical Definition

For finite strain (default):
```math
F = I + ∇u = I + ∑ᵢ uᵢ ⊗ (∂Nᵢ/∂X)
```

where:
- I = identity tensor
- uᵢ = displacement at node i (Vec{3})
- ∂Nᵢ/∂X = basis function derivative w.r.t. material coordinates
- ∂Nᵢ/∂X = J⁻ᵀ ⋅ (∂Nᵢ/∂ξ)

For small strain:
```math
F = I
```

# Arguments
- `X_nodes`: Material coordinates at element nodes (NTuple of Vec{3})
- `u_nodes`: Displacements at element nodes (NTuple of Vec{3})
- `dN_dξ`: Basis function derivatives w.r.t. parametric coords ξ (NTuple of Vec{D})
- `J`: Jacobian matrix ∂X/∂ξ (Tensor{2,3})
- `formulation`: Strain formulation (FiniteStrain() or SmallStrain())

# Returns
- `F::Tensor{2, 3, Float64, 9}`: 3×3 deformation gradient tensor

# Performance
- **Zero allocations**: All operations on stack
- **Type-stable**: Return type known at compile time
- **SIMD-friendly**: Tensors.jl operations vectorize well

# Examples
```julia
# Finite strain (default)
F = compute_deformation_gradient(X_nodes, u_nodes, dN_dξ, J)

# Small strain
F = compute_deformation_gradient(X_nodes, u_nodes, dN_dξ, J, SmallStrain())

# Check properties
@assert det(F) > 0  # Physical requirement
```

# Implementation Notes

1. **Nodal Pattern**: Operates on single element's data (not global vectors)
2. **Tensors.jl**: All math uses Vec and Tensor types
3. **No Mutation**: Pure function, no side effects
4. **GPU-Ready**: Works on GPU without modification
"""
@inline function compute_deformation_gradient(
    X_nodes::NTuple{N,Vec{3,Float64}},
    u_nodes::NTuple{N,Vec{3,Float64}},
    dN_dξ::NTuple{N,Vec{D,Float64}},
    J::Tensor{2,3,Float64,9},
    formulation::StrainFormulation=FiniteStrain()
) where {N,D}

    # Compute Jacobian inverse transpose: J⁻ᵀ = (J⁻¹)ᵀ
    # This maps parametric derivatives to material derivatives:
    # ∂Nᵢ/∂X = J⁻ᵀ ⋅ (∂Nᵢ/∂ξ)
    J_inv = inv(J)
    J_inv_T = transpose(J_inv)

    # Compute displacement gradient: ∇u = ∑ᵢ uᵢ ⊗ (∂Nᵢ/∂X)
    # Start with zero tensor
    grad_u = zero(Tensor{2,3,Float64,9})

    # Accumulate contributions from each node
    @inbounds for i in 1:N
        # Transform parametric derivative to material derivative
        dN_dX = J_inv_T ⋅ dN_dξ[i]

        # Outer product: uᵢ ⊗ (∂Nᵢ/∂X)
        # This is a 3×3 tensor: grad_u[a,b] = u[a] * dN_dX[b]
        grad_u += u_nodes[i] ⊗ dN_dX
    end

    # Compute F based on formulation
    return _apply_formulation(grad_u, formulation)
end

"""
    _apply_formulation(grad_u, ::FiniteStrain)

Apply finite strain formulation: F = I + ∇u
"""
@inline function _apply_formulation(grad_u::Tensor{2,3,Float64,9}, ::FiniteStrain)
    # F = I + ∇u
    return one(Tensor{2,3,Float64,9}) + grad_u
end

"""
    _apply_formulation(grad_u, ::SmallStrain)

Apply small strain formulation: F = I (ignore displacement gradient)
"""
@inline function _apply_formulation(grad_u::Tensor{2,3,Float64,9}, ::SmallStrain)
    # F = I (linearized assumption)
    return one(Tensor{2,3,Float64,9})
end

# High-level API commented out - requires full JuliaFEM Element types
# Uncomment when integrating into main package
#=
"""
    compute_deformation_gradient(
        element::Element{N, NIP, F, B},
        ip_index::Int,
        X_global::Dict{UInt, Vec{3, Float64}},
        u_global::Dict{UInt, Vec{3, Float64}},
        formulation::StrainFormulation = FiniteStrain()
    ) -> Tensor{2, 3, Float64, 9}

High-level API: Compute deformation gradient from element and global state.

This is a convenience wrapper that extracts nodal data and calls the low-level function.

# Arguments
- `element`: Finite element
- `ip_index`: Integration point index (1-based)
- `X_global`: Global material coordinates (node_id → position)
- `u_global`: Global displacements (node_id → displacement)
- `formulation`: Strain formulation (default: FiniteStrain())

# Returns
- `F::Tensor{2, 3, Float64, 9}`: Deformation gradient at integration point

# Examples
```julia
# Setup
X = Dict(UInt(1) => Vec(0.0, 0.0, 0.0), UInt(2) => Vec(1.0, 0.0, 0.0), ...)
u = Dict(UInt(1) => Vec(0.1, 0.0, 0.0), UInt(2) => Vec(0.15, 0.02, 0.0), ...)
element = Element(Tet10, (1,2,3,4,5,6,7,8,9,10))

# Compute F at first integration point
F = compute_deformation_gradient(element, 1, X, u)
```
"""
function compute_deformation_gradient(
    element::Element{N,NIP,F,B},
    ip_index::Int,
    X_global::Dict{UInt,Vec{3,Float64}},
    u_global::Dict{UInt,Vec{3,Float64}},
    formulation::StrainFormulation=FiniteStrain()
) where {N,NIP,F,B}

    # Extract nodal coordinates and displacements
    X_nodes = tuple((X_global[node_id] for node_id in element.connectivity)...)
    u_nodes = tuple((u_global[node_id] for node_id in element.connectivity)...)

    # Get integration point
    ip = element.integration_points[ip_index]
    ξ = Vec(ip.ξ)

    # Get basis function derivatives w.r.t. parametric coordinates
    # Using new API: get_basis_derivatives returns tuple of Vec
    basis_type = typeof(element.basis)
    topology_type = _extract_topology_type(basis_type)
    dN_dξ = get_basis_derivatives(topology_type(), element.basis, ξ)

    # Compute Jacobian: J = ∂X/∂ξ = ∑ᵢ Xᵢ ⊗ (∂Nᵢ/∂ξ)
    J = zero(Tensor{2,3,Float64,9})
    @inbounds for i in 1:N
        J += X_nodes[i] ⊗ dN_dξ[i]
    end

    # Call low-level function
    return compute_deformation_gradient(X_nodes, u_nodes, dN_dξ, J, formulation)
end

"""
    _extract_topology_type(::Type{Lagrange{T, Order}}) where {T, Order}

Extract topology type from Lagrange basis type.
"""
@inline _extract_topology_type(::Type{Lagrange{T,Order}}) where {T,Order} = T
=#

