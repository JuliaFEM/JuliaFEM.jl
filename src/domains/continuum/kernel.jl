# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Continuum mechanics kernel - defines the weak form only.

This module defines:
1. The ContinuumKernel type
2. The weak form: compute_block_at_point (atomic operation)

Everything else (geometry preprocessing, integration, assembly, DOF mapping) belongs elsewhere.
"""

using Tensors

"""
    ContinuumKernel{Theory<:AbstractContinuumTheory, Mat<:AbstractMaterial} <: AbstractKernel

Domain kernel for continuum mechanics (3D solid mechanics).

Couples formulation theory, material model, and displacement field.
Works with any assembler (COO, CSC, nodal).

# Type Parameters
- `Theory`: Continuum theory (FullThreeD, PlaneStress, PlaneStrain, Axisymmetric)
- `Mat`: Material model (LinearElastic, NeoHookean, etc.)

# Fields
- `formulation`: ContinuumFormulation{Theory}
- `material`: Material model instance
- `field`: Displacement{3}() field type

# Example

```julia
kernel = ContinuumKernel(
    ContinuumFormulation{FullThreeD}(),
    LinearElastic(E=210e9, ν=0.3),
    Displacement{3}()
)
```
"""
struct ContinuumKernel{Theory<:AbstractContinuumTheory,Mat<:AbstractMaterial} <: AbstractKernel
    formulation::ContinuumFormulation{Theory}
    material::Mat
    field::Displacement{3}
end

# Convenience constructor without field (defaults to Displacement{3})
function ContinuumKernel(
    formulation::ContinuumFormulation{Theory},
    material::Mat
) where {Theory<:AbstractContinuumTheory,Mat<:AbstractMaterial}
    return ContinuumKernel(formulation, material, Displacement{3}())
end

# ============================================================================
# FIELD ACCESS (for DOF mapping delegation)
# ============================================================================

"""
    get_field(kernel::ContinuumKernel) -> Displacement{3}

Return the field associated with this kernel.

DOF mapping should use dofs_per_node(get_field(kernel)) and similar field-based
functions, not kernel-specific methods.
"""
get_field(kernel::ContinuumKernel) = kernel.field

# ============================================================================
# WEAK FORM - The heart of continuum mechanics
# ============================================================================

"""
    compute_block_at_point(
        grad_k::Vec{3},
        grad_l::Vec{3},
        C::SymmetricTensor{4,3}
    ) -> Tensor{2,3}

Compute the weak form contribution at a single integration point.

This is the **atomic kernel operation** - pure weak form math, no geometry, no loops.
Given shape function gradients and material tensor, compute the 3×3 stiffness block.

# Weak Form

For displacement field u, the weak form of linear momentum is:

```
∫_Ω δε : C : ε dV = ∫_Ω δu ⋅ b dV + ∫_∂Ω δu ⋅ t dS
```

where:
- ε = ½(∇u + (∇u)ᵀ) is the strain (symmetric part of displacement gradient)
- C is the 4th-order material stiffness tensor
- b is body force, t is surface traction

Discretizing u = ∑ Nᵢ uᵢ, the stiffness matrix coupling nodes k and l is:

```
K[k,l][α,β] = ∫_Ω Bₖ,α : C : Bₗ,β dV
```

where Bₖ,α = ½(∇Nₖ ⊗ eα + eα ⊗ ∇Nₖ) is the strain-displacement matrix.

This function computes the integrand (before multiplying by detJ*w).

# Arguments
- `grad_k`: Physical gradient ∇Nₖ at integration point
- `grad_l`: Physical gradient ∇Nₗ at integration point
- `C`: Material stiffness tensor (from elasticity_tensor(material) or compute_stress)

# Returns
3×3 stiffness block K[k,l] at this integration point (before detJ*w scaling)

# Performance
Zero allocations - all tensors stack-allocated.

# Example

```julia
# At an integration point:
grad_k = Vec{3}((0.1, 0.2, 0.3))
grad_l = Vec{3}((0.4, 0.5, 0.6))
C = elasticity_tensor(LinearElastic(E=210e9, ν=0.3))

# Compute weak form contribution
K_kl_ip = compute_block_at_point(grad_k, grad_l, C)

# Integrate: K[k,l] += K_kl_ip * detJ * weight
```
"""
@inline function compute_block_at_point(
    grad_k::Vec{3,Float64},
    grad_l::Vec{3,Float64},
    C::SymmetricTensor{4,3,Float64,36}
)
    # Basis vectors for displacement components
    e_1 = Vec{3}((1.0, 0.0, 0.0))
    e_2 = Vec{3}((0.0, 1.0, 0.0))
    e_3 = Vec{3}((0.0, 0.0, 1.0))
    e = (e_1, e_2, e_3)

    K_kl_ip = zero(Tensor{2,3,Float64,9})

    # Loop over displacement components (α, β ∈ {x, y, z})
    @inbounds for α in 1:3, β in 1:3
        e_α, e_β = e[α], e[β]

        # Strain-displacement B-matrices (symmetric part of displacement gradient)
        # Bₖ,α = ½(∇Nₖ ⊗ eα + eα ⊗ ∇Nₖ)
        B_k_α = 0.5 * (grad_k ⊗ e_α + e_α ⊗ grad_k)
        B_l_β = 0.5 * (grad_l ⊗ e_β + e_β ⊗ grad_l)

        # Weak form: K[α,β] = Bₖ,α : C : Bₗ,β
        # (double contraction of 2nd-order tensors with 4th-order material tensor)
        k_αβ = dcontract(B_k_α, dcontract(C, B_l_β))

        # Accumulate to 3×3 block
        K_kl_ip += k_αβ * (e_α ⊗ e_β)
    end

    return K_kl_ip
end
