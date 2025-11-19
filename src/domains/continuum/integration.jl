# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Continuum mechanics integration utilities.

Provides geometry preprocessing and integration wrappers for the weak form kernel.
These functions are generic and work with any kernel that implements compute_block_at_point.
"""

using StaticArrays
using Tensors

# ============================================================================
# GEOMETRY PREPROCESSING
# ============================================================================

"""
    PreparedElement{N,NIP,GradType,WeightType}

Precomputed element geometry for block-oriented assembly.

Stores all Jacobian-dependent data so blocks can be computed without
recomputing shape function gradients. Created once per element by
`prepare_element!`, then passed to integration functions multiple times.

# Type Parameters
- `N`: Number of nodes in element
- `NIP`: Number of integration points
- `GradType`: Type of gradient storage (NTuple of SVectors)
- `WeightType`: Type of integration weight storage (SVector)

# Fields
- `X`: Node coordinates [N × Vec{3}] (stack-allocated SVector)
- `∇N_data`: Physical gradients at each IP [NIP × (N × Vec{3})]
- `detJ_w`: detJ * weight at each IP [NIP]

# Zero-Allocation
All fields use stack-allocated StaticArrays (SVector, NTuple).
Size known at compile time → perfect type stability.
"""
struct PreparedElement{N,NIP,GradType,WeightType}
    X::SVector{N,Vec{3,Float64}}
    ∇N_data::GradType   # NTuple{NIP, SVector{N, Vec{3}}}
    detJ_w::WeightType  # SVector{NIP, Float64}
end

"""
    prepare_element!(
        cache::ElementCache,
        kernel::ContinuumKernel,
        element_id::Int,
        mesh::AbstractMesh
    ) -> PreparedElement

Precompute element geometry for integration **once**.

Computes:
- Node coordinates (from mesh)
- Physical gradients ∇N at each integration point
- Jacobian determinant × weight (detJ * w) at each integration point

Returned `PreparedElement` can be passed to integration functions multiple times
without recomputing geometry. Essential for nodal assemblers where each node
queries multiple blocks from the same element.

# Arguments
- `cache`: Element cache (provides topology, basis, integration points)
- `kernel`: Continuum kernel
- `element_id`: Element index in mesh
- `mesh`: Finite element mesh

# Returns
`PreparedElement{N,NIP}` with precomputed geometry (stack-allocated)

# Zero-Allocation
Returns immutable struct with SVector/NTuple fields → stack-only, zero heap.
"""
@inline function prepare_element!(
    cache::ElementCache{T,B,IPS},
    kernel::ContinuumKernel,
    element_id::Int,
    mesh::AbstractMesh
) where {T<:AbstractTopology{N},B,IPS} where {N}

    conn = mesh.connectivity[element_id]

    # Load coordinates into SVector (stack-allocated, size N known at compile time)
    X = SVector{N}(ntuple(i -> Vec{3}(mesh.nodes[conn[i]]), N))

    ips = cache.ips
    NIP = length(ips)

    # Precompute physical gradients at all integration points
    # Use ntuple for compile-time size (returns NTuple → stack-allocated)
    ∇N_data = ntuple(NIP) do ip_idx
        ip = ips[ip_idx]
        ξ = Vec{3}(ip.ξ)

        # Reference gradients
        dN_dξ = get_basis_derivatives(cache.topology, cache.basis, ξ)

        # Jacobian: J = X ⊗ ∇_ξ N
        J = X[1] ⊗ dN_dξ[1]
        @inbounds for i in 2:N
            J += X[i] ⊗ dN_dξ[i]
        end

        J_inv_T = transpose(inv(J))

        # Physical gradients for all nodes: ∇N = J^{-T} ⋅ ∇_ξ N
        SVector{N}(ntuple(k -> J_inv_T ⋅ dN_dξ[k], N))
    end

    # Precompute detJ * weight at each integration point
    detJ_w_data = SVector{NIP}(ntuple(NIP) do ip_idx
        ip = ips[ip_idx]
        ξ = Vec{3}(ip.ξ)
        dN_dξ = get_basis_derivatives(cache.topology, cache.basis, ξ)

        J = X[1] ⊗ dN_dξ[1]
        @inbounds for i in 2:N
            J += X[i] ⊗ dN_dξ[i]
        end

        det(J) * ip.weight
    end)

    return PreparedElement{N,NIP,typeof(∇N_data),typeof(detJ_w_data)}(X, ∇N_data, detJ_w_data)
end

# ============================================================================
# INTEGRATION WRAPPERS (generic, trait-based)
# ============================================================================

"""
    compute_tangent_at_point(
        behavior::StatelessConstantTangent,
        material,
        prepared,
        q,
        u_elem,
        state_old,
        Δt
    ) -> 𝔻

Compute tangent modulus for materials with constant tangent.

For constant tangent materials (e.g., LinearElastic), tangent is independent
of strain and integration point. Computed once at reference strain.
"""
@inline function compute_tangent_at_point(
    ::StatelessConstantTangent,
    material,
    prepared,
    q::Int,
    u_elem,
    state_old,
    Δt
)
    # Constant tangent - compute at reference strain
    E_ref = zero(SymmetricTensor{2,3,Float64})
    _, 𝔻, _ = compute_stress(material, E_ref, nothing, 0.0)
    return 𝔻
end

"""
    compute_tangent_at_point(
        behavior::StatelessStrainDependent,
        material,
        prepared,
        q,
        u_elem,
        state_old,
        Δt
    ) -> 𝔻

Compute tangent modulus for materials with strain-dependent tangent.

For strain-dependent materials (e.g., NeoHookean), tangent depends on
deformation state at each integration point. Computes strain from
displacement field and queries material.
"""
@inline function compute_tangent_at_point(
    ::StatelessStrainDependent,
    material,
    prepared::PreparedElement{N},
    q::Int,
    u_elem::AbstractVector{Float64},
    state_old,
    Δt
) where N
    # Compute strain at this integration point
    I = one(Tensor{2,3,Float64})
    ∇N_q = prepared.∇N_data[q]

    # Deformation gradient: F = I + ∇u
    F = I
    @inbounds for k in 1:N
        k_offset = 3(k - 1)
        u_k = Vec{3}((u_elem[k_offset+1], u_elem[k_offset+2], u_elem[k_offset+3]))
        F += u_k ⊗ ∇N_q[k]
    end

    # Green-Lagrange strain: E = ½(C - I) = ½(F'F - I)
    C_tensor = symmetric(F' ⋅ F)
    E = SymmetricTensor{2,3}(0.5 * (C_tensor - I))

    # Get strain-dependent tangent
    _, 𝔻, _ = compute_stress(material, E, nothing, 0.0)
    return 𝔻
end

"""
    compute_tangent_at_point(
        behavior::StatefulStrainDependent,
        material,
        prepared,
        q,
        u_elem,
        state_old,
        Δt
    ) -> (𝔻, state_new)

Compute tangent modulus and update state for stateful materials.

For stateful materials (e.g., PerfectPlasticity), tangent depends on
strain and internal state. Updates state variables during computation.

# Returns
- `𝔻`: Material tangent modulus
- `state_new`: Updated material state at this integration point
"""
@inline function compute_tangent_at_point(
    ::StatefulStrainDependent,
    material,
    prepared::PreparedElement{N},
    q::Int,
    u_elem::AbstractVector{Float64},
    state_old,
    Δt
) where N
    # Compute strain at this integration point (small strain for plasticity)
    I = one(Tensor{2,3,Float64})
    ∇N_q = prepared.∇N_data[q]

    # Small strain: ε = sym(∇u)
    ε = zero(SymmetricTensor{2,3,Float64})
    @inbounds for k in 1:N
        k_offset = 3(k - 1)
        u_k = Vec{3}((u_elem[k_offset+1], u_elem[k_offset+2], u_elem[k_offset+3]))
        ε += symmetric(u_k ⊗ ∇N_q[k])
    end

    # Get state at this integration point (if provided)
    state_q = state_old === nothing ? nothing : state_old[q]

    # Compute stress, tangent, and updated state
    _, 𝔻, state_new = compute_stress(material, ε, state_q, Δt)

    return 𝔻, state_new
end

"""
    compute_block!(
        prepared::PreparedElement,
        material::AbstractMaterial,
        k_local::Int,
        l_local::Int,
        u_elem::AbstractVector{Float64} = Float64[],
        state_old = nothing,
        Δt::Float64 = 0.0
    ) -> Tensor{2,3}

**Generic** integration function for **all materials**.

Integrates weak form over element to get stiffness block K[k,l] between
nodes k and l. Uses material behavior traits to dispatch to appropriate
tangent computation.

# Arguments
- `prepared`: Precomputed element geometry
- `material`: Any material (LinearElastic, NeoHookean, PerfectPlasticity, etc.)
- `k_local`, `l_local`: Local node indices
- `u_elem`: Element displacement DOFs [3N] (optional, needed for strain-dependent materials)
- `state_old`: Material state (optional, needed for stateful materials)
- `Δt`: Time increment (optional, for rate-dependent materials)

# Returns
Fully integrated 3×3 stiffness block K[k,l]

# Performance
Uses trait-based dispatch on `material_behavior(material)`:
- `StatelessConstantTangent`: Tangent computed once, O(1) material queries
- `StatelessStrainDependent`: Tangent computed at each IP, O(NIP) queries
- `StatefulStrainDependent`: Tangent + state update at each IP, O(NIP) queries

# Examples
```julia
# Linear elastic (no u_elem needed)
K_kl = compute_block!(prepared, LinearElastic(E=210e9, ν=0.3), 1, 2)

# Nonlinear elastic (needs u_elem)
K_kl = compute_block!(prepared, NeoHookean(μ=1e6, λ=1e9), 1, 2, u_elem)

# Plastic (needs u_elem and state)
K_kl = compute_block!(prepared, PerfectPlasticity(...), 1, 2, u_elem, state_old, Δt)
```
"""
@inline function compute_block!(
    prepared::PreparedElement{N,NIP},
    material::AbstractMaterial,
    k_local::Int,
    l_local::Int,
    u_elem::AbstractVector{Float64} = Float64[],
    state_old = nothing,
    Δt::Float64 = 0.0
) where {N,NIP}

    behavior = material_behavior(material)
    K_kl = zero(Tensor{2,3,Float64})

    # Optimization: For constant tangent, compute once and reuse
    if behavior isa StatelessConstantTangent
        𝔻 = compute_tangent_at_point(behavior, material, prepared, 1, u_elem, state_old, Δt)

        @inbounds for q in 1:NIP
            grad_k = prepared.∇N_data[q][k_local]
            grad_l = prepared.∇N_data[q][l_local]
            K_kl_ip = compute_block_at_point(grad_k, grad_l, 𝔻)
            K_kl += K_kl_ip * prepared.detJ_w[q]
        end
    else
        # Strain-dependent: compute tangent at each integration point
        @inbounds for q in 1:NIP
            # Dispatch handles both stateless and stateful cases
            if behavior isa StatefulStrainDependent
                𝔻, _ = compute_tangent_at_point(behavior, material, prepared, q, u_elem, state_old, Δt)
            else
                𝔻 = compute_tangent_at_point(behavior, material, prepared, q, u_elem, state_old, Δt)
            end

            grad_k = prepared.∇N_data[q][k_local]
            grad_l = prepared.∇N_data[q][l_local]
            K_kl_ip = compute_block_at_point(grad_k, grad_l, 𝔻)
            K_kl += K_kl_ip * prepared.detJ_w[q]
        end
    end

    return K_kl
end

# ============================================================================
# BACKWARD COMPATIBILITY (element-based assemblers)
# ============================================================================

"""
    compute_all_blocks!(
        K_blocks::AbstractMatrix{Tensor{2,3}},
        prepared::PreparedElement,
        material::AbstractMaterial,
        u_elem::AbstractVector{Float64},
        Nnodes::Int,
        state_old = nothing,
        Δt::Float64 = 0.0
    )

**Generic** function to compute all N×N stiffness blocks for **any material**.

Helper for element-based assemblers. For nodal assemblers, call compute_block!
directly for only the needed blocks.

# Arguments
- `K_blocks`: Output matrix [N×N] of 3×3 tensor blocks
- `prepared`: Precomputed element geometry
- `material`: Any material (LinearElastic, NeoHookean, PerfectPlasticity, etc.)
- `u_elem`: Element displacement DOFs [3N]
- `Nnodes`: Number of nodes in element
- `state_old`: Material state (optional, for stateful materials)
- `Δt`: Time increment (optional, for rate-dependent materials)

# Examples
```julia
# Linear elastic
compute_all_blocks!(K_blocks, prepared, LinearElastic(E=210e9, ν=0.3), u_elem, 8)

# Nonlinear elastic
compute_all_blocks!(K_blocks, prepared, NeoHookean(μ=1e6, λ=1e9), u_elem, 8)

# Plastic
compute_all_blocks!(K_blocks, prepared, PerfectPlasticity(...), u_elem, 8, state_old, Δt)
```
"""
@inline function compute_all_blocks!(
    K_blocks::AbstractMatrix{<:Tensor{2,3}},
    prepared::PreparedElement{N},
    material::AbstractMaterial,
    u_elem::AbstractVector{Float64},
    Nnodes::Int,
    state_old = nothing,
    Δt::Float64 = 0.0
) where {N}
    @inbounds for k in 1:Nnodes, l in 1:Nnodes
        K_blocks[k, l] = compute_block!(prepared, material, k, l, u_elem, state_old, Δt)
    end
end

"""
    blocked_tensor_to_matrix_view!(
        K_e::AbstractMatrix{Float64},
        K_blocks::AbstractMatrix{Tensor{2,3}}
    )

Convert N×N matrix of 3×3 tensor blocks to 3N×3N Float64 matrix.

Maps block[k,l][α,β] → K_e[3(k-1)+α, 3(l-1)+β]
"""
function blocked_tensor_to_matrix_view!(
    K_e::AbstractMatrix{Float64},
    K_blocks::AbstractMatrix{<:Tensor{2,3}}
)
    Nnodes = size(K_blocks, 1)
    @inbounds for k in 1:Nnodes, l in 1:Nnodes
        block = K_blocks[k, l]
        k_offset = 3(k - 1)
        l_offset = 3(l - 1)
        for α in 1:3, β in 1:3
            K_e[k_offset + α, l_offset + β] = block[α, β]
        end
    end
end

"""
    compute_element_stiffness!(
        cache::ElementCache,
        kernel::ContinuumKernel,
        element_id::Int,
        mesh::AbstractMesh
    )

Compute element stiffness matrix and force vector (writes to cache).

Implements the kernel interface for element-based assemblers (COO, CSC).
Writes results to `cache.Ke` and `cache.fe` without allocating.

For nodal assemblers, use prepare_element! + compute_block! directly.

# Arguments
- `cache`: Element cache with pre-allocated Ke, fe, K_blocks
- `kernel`: Continuum kernel with material
- `element_id`: Element index in mesh
- `mesh`: Finite element mesh

# Side Effects
Writes to:
- `cache.Ke` - Element stiffness matrix [3N × 3N]
- `cache.fe` - Element force vector [3N] (zeros for LinearElastic)
"""
function compute_element_stiffness!(
    cache::ElementCache{T,B,IPS},
    kernel::ContinuumKernel{M},
    element_id::Int,
    mesh::AbstractMesh
) where {T<:AbstractTopology{N},B,IPS,M} where {N}

    # Zero outputs
    fill!(cache.Ke, 0.0)
    fill!(cache.fe, 0.0)

    # Prepare element geometry (uses cache.X_buffer for coordinates)
    prepared = prepare_element!(cache, kernel, element_id, mesh)

    # Compute all blocks into cache.K_blocks (reuses existing allocation)
    compute_all_blocks!(cache.K_blocks, prepared, kernel.material, cache.u_buffer, N)

    # Convert blocks to Float64 matrix in cache.Ke
    blocked_tensor_to_matrix_view!(cache.Ke, cache.K_blocks)

    return nothing
end
