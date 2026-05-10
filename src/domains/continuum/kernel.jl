# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

"""
Continuum mechanics kernel - defines the weak form only.

This module defines:
1. The ContinuumKernel type
2. The weak form: compute_stiffness_value (most atomic operation)
3. Block builder: compute_stiffness_block (builds D×D blocks)

The DOF-based / matrix-free assembler microkernel surface
(`qpoint_buffer_eltype`, `prepare_dof_based_material_workspace!`,
`update_qpoint_buffer!`, `evaluate_entry`, `evaluate_mass_entry`,
`reference_fields`) is implemented in this file and in
`dof_based_pass1.jl`. Everything else (geometry preprocessing,
integration, assembly, DOF mapping) belongs elsewhere.
"""

using Tensors
using Tensors: basevec  # For unit vector construction

"""
    ContinuumKernel{Theory<:AbstractContinuumTheory, Mat<:AbstractMaterial} <: AbstractKernel

Domain kernel for continuum mechanics (3D solid mechanics).

Couples formulation theory, material model, displacement field, and
optional density (carried on the kernel rather than the material so
existing material structs stay untouched).

# Type Parameters
- `Theory`: Continuum theory (ThreeDimensional, PlaneStress, PlaneStrain, Axisymmetric)
- `Mat`: Material model (LinearElastic, NeoHookean, etc.)

# Fields
- `formulation`: ContinuumFormulation{Theory}
- `material`: Material model instance
- `field`: Displacement{3}() field type
- `density::Float64`: mass density `ρ` [kg/m³] used by the mass matrix
  (`evaluate_mass_entry` / `apply_M!` / `assemble_M!`). Defaults to `0`,
  in which case the kernel produces a structural-zero `M` and the
  static-elasticity tests are unchanged.

# Example

```julia
kernel = ContinuumKernel(
    ContinuumFormulation{ThreeDimensional}(),
    LinearElastic(E=210e9, ν=0.3),
    Displacement{3}();
    density = 7850.0,        # for mass matrix; omit for static-only
)
```
"""
struct ContinuumKernel{Theory<:AbstractContinuumTheory,Mat<:AbstractMaterial} <: AbstractKernel
    formulation::ContinuumFormulation{Theory}
    material::Mat
    field::Displacement{3}
    density::Float64
end

# Outer constructor: positional 3-arg form, density defaults to 0.
function ContinuumKernel(
    formulation::ContinuumFormulation{Theory},
    material::Mat,
    field::Displacement{3};
    density::Float64 = 0.0,
) where {Theory<:AbstractContinuumTheory,Mat<:AbstractMaterial}
    return ContinuumKernel{Theory, Mat}(formulation, material, field, density)
end

# Convenience constructor without field (defaults to Displacement{3}).
function ContinuumKernel(
    formulation::ContinuumFormulation{Theory},
    material::Mat;
    density::Float64 = 0.0,
) where {Theory<:AbstractContinuumTheory,Mat<:AbstractMaterial}
    return ContinuumKernel{Theory, Mat}(formulation, material, Displacement{3}(), density)
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

Compute single scalar stiffness value K[k,l][α,β] at integration point.

This is the most atomic kernel operation - computes one DOF-pair contribution.

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
    grad_k::Vec{D,F},
    grad_l::Vec{D,F},
    C::Tensor{4,D,F},
    α::Int,
    β::Int
) where {D,F<:AbstractFloat}
    # Build strain-displacement operators Bₖ,α and Bₗ,β
    # These are 2nd-order tensors (D×D matrices)

    # eα and eβ are unit vectors of the same float type as the inputs,
    # so the symmetrization below stays at precision F end-to-end.
    e_α = basevec(Vec{D,F}, α)
    e_β = basevec(Vec{D,F}, β)

    half = F(0.5)

    # Bₖ,α = ½(∇Nₖ ⊗ eα + eα ⊗ ∇Nₖ)
    B_k_α = half * (grad_k ⊗ e_α + e_α ⊗ grad_k)

    # Bₗ,β = ½(∇Nₗ ⊗ eβ + eβ ⊗ ∇Nₗ)
    B_l_β = half * (grad_l ⊗ e_β + e_β ⊗ grad_l)

    # Compute weak form: K[α,β] = Bₖ,α : C : Bₗ,β
    # Double contraction: sum over all indices
    return dcontract(B_k_α, dcontract(C, B_l_β))
end

"""
    compute_stiffness_value(
        grad_k::Vec{D,F},
        grad_l::Vec{D,F},
        C::SymmetricTensor{4,D,F},
        α::Int, β::Int,
    ) -> F

`SymmetricTensor` overload of the atomic stiffness microkernel — computes
the same `B_k,α : C : B_l,β` value but without converting `C` from
`SymmetricTensor` to the full `Tensor`. The full conversion goes
through `Tensors.jl`'s general `Tensor{4,3}(::SymmetricTensor)`
constructor, which carries an error-string branch that pulls in
`Base.string` / `print_to_string` — call sites the Metal codegen can't
prove dead and which trigger a `julia.new_gc_frame` IR error during
device compilation.

Building `B_k,α` and `B_l,β` as `SymmetricTensor`s via `symmetric(...)`
keeps the entire chain inside the symmetric-tensor methods of
`dcontract`, all of which are GPU-clean. Bit-identical to the
`Tensor`-based variant on the CPU.
"""
@inline function compute_stiffness_value(
    grad_k::Vec{D,F},
    grad_l::Vec{D,F},
    C::SymmetricTensor{4,D,F},
    α::Int,
    β::Int
) where {D,F<:AbstractFloat}
    e_α = basevec(Vec{D,F}, α)
    e_β = basevec(Vec{D,F}, β)
    half = F(0.5)
    B_k_α = symmetric(half * (grad_k ⊗ e_α + e_α ⊗ grad_k))
    B_l_β = symmetric(half * (grad_l ⊗ e_β + e_β ⊗ grad_l))
    return dcontract(B_k_α, dcontract(C, B_l_β))
end

"""
    compute_internal_force_value(grad_i::Vec{3,F}, σ::SymmetricTensor{2,3,F}, α::Int) where {F}

Scalar factor for the Galerkin internal-force row of a displacement test function
associated with shape function ``N_i`` (gradient ``\\nabla N_i``) and Cartesian
component ``\\alpha``:

``\\sigma_{j\\alpha} \\, \\partial N_i / \\partial x_j``

(sum over ``j = 1\\ldots 3``). The caller multiplies by ``\\det J \\cdot w`` per
quadrature point and accumulates over IPs and elements.

Cauchy stress ``\\sigma`` is the value stored in the material workspace at the IP
(small-strain or finite-strain model, depending on the constitutive update).

Zero allocation.
"""
@inline function compute_internal_force_value(
    grad_i::Vec{3,F},
    σ::SymmetricTensor{2,3,F},
    α::Int,
) where {F<:AbstractFloat}
    s = zero(F)
    @inbounds for j in 1:3
        s += grad_i[j] * σ[j, α]
    end
    return s
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

Accepts `C` as a `SymmetricTensor{4,3}`; internally uses the
dimension-generic `compute_stiffness_block()`.
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
    compute_block!(
        K_blocks::Matrix{Tensor{2,3,Float64,9}},
        ∇N_data::Matrix{Vec{3,Float64}},
        detJ_w::Vector{Float64},
        𝔻::Vector{SymmetricTensor{4,3,Float64,36}},
        k_local::Int, l_local::Int,
    ) -> Nothing

Integrate one continuum stiffness block `K[k_local, l_local]` over the
quadrature points of an element using the precomputed material tangents
`𝔻`. Writes directly into `K_blocks[k_local, l_local]`.

Used by the element-based COO assembler in
`src/assemblers/element_based/element_based_coo.jl`. The phase-separated
design keeps material evaluations in `update_material_cache!` and lets
this hot loop touch only `Vec{3}` / `SymmetricTensor{4,3}` arithmetic.

Zero-allocation; the inner loop is `@inbounds`.
"""
function compute_block!(
    K_blocks::Matrix{Tensor{2,3,Float64,9}},
    ∇N_data::Matrix{Vec{3,Float64}},
    detJ_w::Vector{Float64},
    𝔻::Vector{SymmetricTensor{4,3,Float64,36}},
    k_local::Int,
    l_local::Int,
)
    K_kl = zero(Tensor{2,3,Float64,9})

    NIP = length(detJ_w)
    @inbounds for q in 1:NIP
        grad_k = ∇N_data[q, k_local]
        grad_l = ∇N_data[q, l_local]
        w = detJ_w[q]
        D = 𝔻[q]

        K_kl_ip = compute_block_at_point(grad_k, grad_l, D)
        K_kl += K_kl_ip * w
    end

    K_blocks[k_local, l_local] = K_kl
    return nothing
end

# ============================================================================
# Microkernel contract for the DOF-based assembler
# ============================================================================
# `qpoint_buffer_eltype`, `update_qpoint_buffer!`, `evaluate_entry` are the
# kernel-agnostic surface defined in `src/assemblers/microkernel.jl`. These
# three methods opt `ContinuumKernel` in.
#
# The per-IP buffer is the elasticity tensor `𝔻` stored as a `SymmetricTensor`,
# which is the only material data the displacement-only weak form needs.

import ..JuliaFEM: qpoint_buffer_eltype, update_qpoint_buffer!, evaluate_entry,
                   evaluate_mass_entry,
                   reference_fields,
                   DOFLayoutEntry, entity_local, component, extract_tangent!,
                   AssemblyMaterialWorkspace, compute_stress
using Tensors: SymmetricTensor

@inline qpoint_buffer_eltype(::ContinuumKernel) = SymmetricTensor{4,3,Float64,36}

"""
    reference_fields(kernel::ContinuumKernel)

The continuum-mechanics weak form needs `(σ, 𝔻)` per IP. For the linear
case both are the constitutive evaluation at zero strain — a one-shot
constant pre-computed here so Pass 1 of the DOF-based assembler can fill
the per-element material workspace by simple copy.
"""
@inline function reference_fields(kernel::ContinuumKernel)
    E_ref = zero(SymmetricTensor{2,3,Float64,6})
    σ_ref, 𝔻_ref, _ = compute_stress(kernel.material, E_ref, NamedTuple(), 0.0)
    return ((σ = σ_ref, 𝔻 = 𝔻_ref), NamedTuple())
end

# `buffer` is `AbstractVector` so the assembler can pass either a plain
# `Vector{Buf}` (legacy) or a column view into a `Matrix{Buf}` (the new
# flattened layout) without going through a copy.
@inline function update_qpoint_buffer!(
    buffer::AbstractVector{SymmetricTensor{4,3,Float64,36}},
    workspace::AssemblyMaterialWorkspace{FieldType, StateType},
    ::ContinuumKernel,
) where {FieldType, StateType}
    fields = getfield(workspace, 1)
    extract_tangent!(buffer, fields, FieldType)
    return nothing
end

"""
    evaluate_entry(kernel::ContinuumKernel, geometry_cache,
                   𝔻_vec::Vector{SymmetricTensor{4,3,Float64,36}},
                   layout_i::DOFLayoutEntry, layout_j::DOFLayoutEntry,
                   elem_id::Int) -> Float64

Continuum-mechanics microkernel for the DOF-based assembler.

Single-field (displacement) so the field index in each `DOFLayoutEntry`
is ignored; only `(entity_local, component)` matter. Sums
`∇Nᵢ : 𝔻 : ∇Nⱼ * detJ·w` over quadrature points and returns the scalar.
The volume kernel ignores `elem_id`.

Allocation-free; the inner integration is `@inbounds`. The actual
`compute_stiffness_value` call is the same atomic kernel that the
element-based assembler also uses, so by construction the two assemblers
must agree to round-off.
"""
@inline function evaluate_entry(
    kernel::ContinuumKernel,
    geometry_cache,
    𝔻_vec::AbstractVector{<:SymmetricTensor{4,3}},
    layout_i::DOFLayoutEntry,
    layout_j::DOFLayoutEntry,
    ::Int,
)
    node_i = entity_local(layout_i)
    comp_i = component(layout_i)
    node_j = entity_local(layout_j)
    comp_j = component(layout_j)

    F = eltype(geometry_cache.detJ_w)
    K_ij = zero(F)
    n_ips = length(geometry_cache.detJ_w)
    @inbounds for q in 1:n_ips
        ∇N_i  = geometry_cache.∇N_data[q, node_i]
        ∇N_j  = geometry_cache.∇N_data[q, node_j]
        detJw = geometry_cache.detJ_w[q]
        # Pass the SymmetricTensor straight through — the SymmetricTensor
        # overload of compute_stiffness_value avoids the
        # `Tensor{4,3}(::SymmetricTensor)` conversion that pulls in
        # `Base.string` and breaks Metal codegen with `julia.new_gc_frame`.
        K_ij += compute_stiffness_value(∇N_i, ∇N_j, 𝔻_vec[q], comp_i, comp_j) * detJw
    end
    return K_ij
end

"""
    evaluate_mass_entry(kernel::ContinuumKernel, geometry_cache, qp_buffer,
                        layout_i, layout_j) -> Float64

Continuum mass matrix microkernel. Returns

    M[i, j] = δ_{α,β} · ρ · Σ_q N_i(q) · N_j(q) · detJ·w(q)

i.e. the *consistent* (not lumped) mass matrix block-diagonal in the
displacement components (`δ_{α,β}` zeros the off-component entries).
Returns `0.0` when `kernel.density == 0`, so kernels constructed without
a density behave exactly as before — callers who only ever assemble `K`
pay no extra cost.

`qp_buffer` is unused in the linear case but kept in the signature so
variable-density materials drop in via a one-line change.
"""
@inline function evaluate_mass_entry(
    kernel::ContinuumKernel,
    geometry_cache,
    qp_buffer,
    layout_i::DOFLayoutEntry,
    layout_j::DOFLayoutEntry,
)
    F = eltype(geometry_cache.detJ_w)
    ρ = F(kernel.density)
    if ρ == zero(F)
        return zero(F)
    end

    comp_i = component(layout_i)
    comp_j = component(layout_j)
    if comp_i != comp_j           # mass matrix is block-diagonal in (α, β)
        return zero(F)
    end

    node_i = entity_local(layout_i)
    node_j = entity_local(layout_j)

    M_ij  = zero(F)
    n_ips = length(geometry_cache.detJ_w)
    @inbounds for q in 1:n_ips
        N_i   = geometry_cache.N_data[q, node_i]
        N_j   = geometry_cache.N_data[q, node_j]
        detJw = geometry_cache.detJ_w[q]
        M_ij += N_i * N_j * detJw
    end
    return ρ * M_ij
end
