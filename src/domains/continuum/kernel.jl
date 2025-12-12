# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Continuum mechanics kernel - defines the weak form only.

This module defines:
1. The ContinuumKernel type
2. The weak form: compute_stiffness_value (most atomic operation)
3. Block builder: compute_stiffness_block (builds D×D blocks)
4. NEW: Microkernel interface via evaluate() wrapper

Everything else (geometry preprocessing, integration, assembly, DOF mapping) belongs elsewhere.
"""

using Tensors
using Tensors: basevec  # For unit vector construction

# Import for microkernel interface and material cache accessors
if isdefined(Main, :JuliaFEM) && isdefined(Main.JuliaFEM, :evaluate)
    import ..JuliaFEM: evaluate, get_tangent
end

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
    compute_stiffness_value(
        grad_k::Vec{D},
        grad_l::Vec{D},
        C::Tensor{4,D},
        α::Int,
        β::Int
    ) -> Float64

Compute **single scalar** stiffness value K[k,l][α,β] at integration point.

This is the **most atomic kernel operation** - computes one DOF-pair contribution.

# Theory

For displacement field u with components uₐ (α = 1,2,3 for 3D), the weak form is:

    K[k,l][α,β] = ∫_Ω Bₖ,α : C : Bₗ,β dV

where:
- Bₖ,α = ½(∇Nₖ ⊗ eα + eα ⊗ ∇Nₖ) = strain-displacement operator
- C = 4th-order elasticity tensor
- eα = unit vector in direction α

Expanded in index notation:
    
    K[k,l][α,β] = ½ C[α,i,β,j] (∂Nₖ/∂xᵢ) (∂Nₗ/∂xⱼ) 
                + ½ C[i,α,β,j] (∂Nₖ/∂xᵢ) (∂Nₗ/∂xⱼ)
                + ½ C[α,i,j,β] (∂Nₖ/∂xᵢ) (∂Nₗ/∂xⱼ)
                + ½ C[i,α,j,β] (∂Nₖ/∂xᵢ) (∂Nₗ/∂xⱼ)

Using symmetry of C, this simplifies to the implementation below.

# Arguments
- `grad_k`: Physical gradient ∇Nₖ at integration point (Vec{D})
- `grad_l`: Physical gradient ∇Nₗ at integration point (Vec{D})
- `C`: Material stiffness tensor (Tensor{4,D})
- `α`: DOF component at node k (1,2,3 for x,y,z)
- `β`: DOF component at node l (1,2,3 for x,y,z)

# Returns
Scalar contribution to K[k,l][α,β] (before detJ*w scaling)

# Performance
Zero allocations, fully inlined, SIMD-friendly.

# Example
```julia
# At integration point:
grad_k = Vec{3}((0.1, 0.2, 0.3))
grad_l = Vec{3}((0.4, 0.5, 0.6))
C = elasticity_tensor(material)

# Compute K[k,l][2,3] (y-component of node k, z-component of node l)
k_23 = compute_stiffness_value(grad_k, grad_l, C, 2, 3)
```

# Design Note
This is more atomic than 3×3 blocks. Caller can:
1. Build blocks: `K_kl[α,β] = compute_stiffness_value(...)` 
2. Direct assembly: `K_global[dof_k_α, dof_l_β] += value * detJ * w`
"""
@inline function compute_stiffness_value(
    grad_k::Vec{D,Float64},
    grad_l::Vec{D,Float64},
    C::Tensor{4,D,Float64},
    α::Int,
    β::Int
) where D
    # Build strain-displacement operators Bₖ,α and Bₗ,β
    # These are 2nd-order tensors (D×D matrices)
    
    # eα and eβ are unit vectors
    e_α = basevec(Vec{D}, α)
    e_β = basevec(Vec{D}, β)
    
    # Bₖ,α = ½(∇Nₖ ⊗ eα + eα ⊗ ∇Nₖ)
    B_k_α = 0.5 * (grad_k ⊗ e_α + e_α ⊗ grad_k)
    
    # Bₗ,β = ½(∇Nₗ ⊗ eβ + eβ ⊗ ∇Nₗ)
    B_l_β = 0.5 * (grad_l ⊗ e_β + e_β ⊗ grad_l)
    
    # Compute weak form: K[α,β] = Bₖ,α : C : Bₗ,β
    # Double contraction: sum over all indices
    return dcontract(B_k_α, dcontract(C, B_l_β))
end

"""
    compute_stiffness_block(
        grad_k::Vec{D},
        grad_l::Vec{D},
        C::Tensor{4,D}
    ) -> Tensor{2,D}

Compute D×D stiffness block K[k,l] at integration point.

This builds a block by calling the atomic `compute_stiffness_value()` kernel.
Use this when you want 3×3 (or 2×2) blocks. Use atomic kernel for direct assembly.

# Arguments
- `grad_k`: Physical gradient ∇Nₖ
- `grad_l`: Physical gradient ∇Nₗ
- `C`: Material stiffness tensor

# Returns
D×D tensor K[k,l] at this integration point (before detJ*w scaling)

# Example
```julia
# 3×3 block for 3D
K_kl = compute_stiffness_block(grad_k, grad_l, C)

# Direct assembly
K_global[3*(k-1)+1:3*k, 3*(l-1)+1:3*l] += K_kl * detJ * w
```
"""
@inline function compute_stiffness_block(
    grad_k::Vec{D,Float64},
    grad_l::Vec{D,Float64},
    C::Tensor{4,D,Float64}
) where D
    # Build D×D block by calling atomic kernel
    K_kl = zero(Tensor{2,D,Float64})
    
    @inbounds for α in 1:D, β in 1:D
        K_kl += compute_stiffness_value(grad_k, grad_l, C, α, β) * 
                (basevec(Vec{D}, α) ⊗ basevec(Vec{D}, β))
    end
    
    return K_kl
end

"""
    compute_block_at_point(grad_k, grad_l, C) -> Tensor{2,3}

Computes 3×3 stiffness block from gradients and elasticity tensor.

Maintains API compatibility with `src/assemblers/kernel_interface.jl` which
uses SymmetricTensor. Internally uses dimension-generic `compute_stiffness_block()`.
"""
@inline function compute_block_at_point(
    grad_k::Vec{3,Float64},
    grad_l::Vec{3,Float64},
    C::SymmetricTensor{4,3,Float64,36}
)
    # Convert SymmetricTensor to Tensor for computation
    C_tensor = Tensor{4,3}(C)
    return compute_stiffness_block(grad_k, grad_l, C_tensor)
end

"""
    evaluate(
        kernel::ContinuumKernel{Theory},
        ::Displacement{3}, ::Displacement{3},
        k::Int, l::Int, α::Int, β::Int,
        material_cache, geometry_cache, q::Int
    ) -> Float64

Microkernel interface for continuum mechanics stiffness assembly.

Wraps `compute_stiffness_value()` to provide dispatch-based field coupling.

# Implementation

Extracts gradients and elasticity tensor from caches, then calls
`compute_stiffness_value()` for the actual computation.

# Example

```julia
# Direct computation
grad_k = geometry_cache.∇N_data[q, k]
grad_l = geometry_cache.∇N_data[q, l]
    @inbounds C = get_tangent(material_workspace, q)
value = compute_stiffness_value(grad_k, grad_l, C, α, β)

# Microkernel interface
value = evaluate(kernel, Displacement{3}(), Displacement{3}(),
                k, l, α, β, material_workspace, geometry_cache, q)
```

# Performance

No overhead - compiler inlines to identical code.
"""
@inline function evaluate(
    kernel::ContinuumKernel{Theory},
    ::Displacement{3}, ::Displacement{3},
    k::Int, l::Int, α::Int, β::Int,
    material_cache, geometry_cache, q::Int
) where {Theory<:AbstractContinuumTheory}
    # Extract from caches using indices
    grad_k = geometry_cache.∇N_data[q, k]
    grad_l = geometry_cache.∇N_data[q, l]
    @inbounds C = get_tangent(material_cache, q)
    
    # Call existing implementation (no duplication!)
    return compute_stiffness_value(grad_k, grad_l, C, α, β)
end
