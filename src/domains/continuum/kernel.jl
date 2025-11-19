# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Continuum mechanics kernel for generic assemblers.

Implements the kernel interface for 3D continuum mechanics (solid mechanics).
Compatible with COOAssembler, CSCAssembler, and future NodalAssembler.
"""

using Tensors
using LinearAlgebra

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

# Integration and Basis

Kernel automatically selects appropriate integration order and basis functions
based on topology type during assembly.

# Example

```julia
kernel = ContinuumKernel(
    ContinuumFormulation{FullThreeD}(),
    LinearElastic(E=210e9, ν=0.3),
    Displacement{3}()
)

# Use with any assembler
assembler = CSCAssembler()
cache = create_cache(assembler, mesh, kernel)
assemble!(cache, assembler, kernel, mesh)
K, f = extract_system(cache)
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
# KERNEL INTERFACE IMPLEMENTATION
# ============================================================================

"""
    dofs_per_node(kernel::ContinuumKernel) -> Int

Continuum mechanics uses 3 DOFs per node (ux, uy, uz).
"""
function dofs_per_node(kernel::ContinuumKernel)
    return 3  # ux, uy, uz displacements
end

"""
    get_dof_mapping!(
        dofs::Vector{Int},
        kernel::ContinuumKernel,
        element_id::Int,
        mesh::AbstractMesh
    ) -> Nothing

Fill DOF indices for continuum element (node-major ordering).

DOF numbering: Node k has DOFs [3*(k-1)+1, 3*(k-1)+2, 3*(k-1)+3] for [ux, uy, uz].

# Example

Element with nodes [10, 20, 30, 40]:
- Node 10: DOFs [28, 29, 30]
- Node 20: DOFs [58, 59, 60]
- Node 30: DOFs [88, 89, 90]
- Node 40: DOFs [118, 119, 120]

Output: dofs = [28, 29, 30, 58, 59, 60, 88, 89, 90, 118, 119, 120]
"""
function get_dof_mapping!(
    dofs::AbstractVector{Int},
    kernel::ContinuumKernel,
    element_id::Int,
    mesh::AbstractMesh
)
    conn = mesh.connectivity[element_id]
    nnodes_elem = length(conn)

    # Fill DOF indices (node-major: all DOFs for node 1, then node 2, ...)
    idx = 1
    @inbounds for node_id in conn
        for α in 1:3  # ux, uy, uz
            dofs[idx] = 3 * (node_id - 1) + α
            idx += 1
        end
    end

    return nothing
end

"""
    compute_element_stiffness!(
        cache::ElementCache,
        kernel::ContinuumKernel,
        element_id::Int,
        mesh::AbstractMesh
    ) -> Nothing

Compute element stiffness matrix and force vector **using block API**.

**NEW IMPLEMENTATION (Nov 2025)**: This is now a thin wrapper over the
composable block API:
1. `prepare_element!` - Precompute geometry once
2. `compute_block!` - Loop over node pairs
3. Convert blocks to Float64 matrix

This design allows:
- Element assemblers: Use this function (full Ke)
- Nodal assemblers: Call `prepare_element!` + `compute_block!` directly
- GPU kernels: Use `compute_block_at_point` (atomic operation)

# Zero-Allocation Guarantee

All computations use stack-allocated types (SVector, NTuple, Tensor).
No heap allocations.

# Arguments
- `cache`: Pre-allocated element workspace
- `kernel`: Continuum kernel with material and formulation
- `element_id`: Element index in mesh
- `mesh`: Finite element mesh
"""
function compute_element_stiffness!(
    cache::ElementCache{T,B,IPS},
    kernel::ContinuumKernel,
    element_id::Int,
    mesh::AbstractMesh
) where {T<:AbstractTopology{N},B,IPS} where {N}

    ndofs_elem = 3 * N

    # Zero output arrays
    @views fill!(cache.Ke[1:ndofs_elem, 1:ndofs_elem], 0.0)
    @views fill!(cache.fe[1:ndofs_elem], 0.0)

    # LEVEL 2: Prepare element geometry ONCE
    prepared = prepare_element!(cache, kernel, element_id, mesh)

    # LEVEL 3: Compute all blocks using material-dispatched block API
    u_elem_view = @view cache.u_buffer[1:ndofs_elem]
    fill!(u_elem_view, 0.0)  # Zero displacements (for linear or initial tangent)

    compute_all_blocks!(cache.K_blocks, prepared, kernel.material, u_elem_view, N)

    # Convert blocked tensor to Float64 matrix (cache.Ke)
    blocked_tensor_to_matrix_view!(
        @view(cache.Ke[1:ndofs_elem, 1:ndofs_elem]),
        @view(cache.K_blocks[1:N, 1:N])
    )

    # TODO: Add body forces to fe if needed
    # For now, fe = 0 (forces added by Neumann BCs)

    return nothing
end

# ============================================================================
# BLOCK-ORIENTED API (Composable functions for nodal and element assemblers)
# ============================================================================
#
# Architecture:
#   Level 1: compute_block_at_point - Single integration point (atomic kernel)
#   Level 2: PreparedElement, prepare_element! - Geometry preprocessing
#   Level 3: compute_block! - Single node-pair integration
#   Level 4: compute_element_stiffness! - Full element (uses Level 3)
#
# This layered design allows:
# - Element assemblers: call Level 4 (full Ke matrix)
# - Nodal assemblers: call Level 2 + Level 3 (individual blocks)
# - GPU kernels: call Level 1 (pure math, perfect for CUDA)
#
# All functions are zero-allocation using stack types (Tensor, Vec, SVector, NTuple)
# ============================================================================

using StaticArrays

"""
    PreparedElement{N,NIP,GradType,WeightType}

Precomputed element geometry for block-oriented assembly.

Stores all Jacobian-dependent data so blocks can be computed without
recomputing shape function gradients. Created once per element by
`prepare_element!`, then passed to `compute_block!` multiple times.

# Type Parameters
- `N`: Number of nodes in element
- `NIP`: Number of integration points
- `GradType`: Type of gradient storage (SVector of physical gradients)
- `WeightType`: Type of integration weight storage (SVector of detJ*w)

# Fields
- `X`: Node coordinates [N × Vec{3}] (stack-allocated SVector)
- `∇N_data`: Physical gradients at each IP [NIP × (N × Vec{3})]
- `detJ_w`: detJ * weight at each IP [NIP]

# Zero-Allocation
All fields use stack-allocated StaticArrays (SVector, NTuple).
Size known at compile time → perfect type stability.

# Usage (Nodal Assembler)

```julia
# Prepare element once
prepared = prepare_element!(cache, kernel, elem_id, mesh)

# Query blocks many times (zero recomputation)
for local_j in 1:nnodes_elem
    block = compute_block!(prepared, material, local_i, local_j)
    # ... accumulate to row
end
```
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

Precompute element geometry for block assembly **once**.

Computes:
- Node coordinates (from mesh)
- Physical gradients ∇N at each integration point
- Jacobian determinant × weight (detJ * w) at each integration point

Returned `PreparedElement` can be passed to `compute_block!` multiple times
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

# Performance
- **Nodal assembler**: Prepare once per element, query N² blocks
- **Element assembler**: Prepare once, build full Ke via block API
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

    # Precompute physical gradients and detJ*w at all integration points
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

"""
    compute_block_at_point(
        grad_k::Vec{3},
        grad_l::Vec{3},
        C::SymmetricTensor{4,3}
    ) -> Tensor{2,3}

Compute 3×3 stiffness block at **single integration point** (before scaling by detJ*w).

This is the **atomic kernel operation** - pure tensor math, no geometry, no loops.
Perfect for:
- GPU CUDA kernels (SIMD-friendly)
- CPU vectorization
- Maximum code reuse

# Algorithm
For displacement DOFs α,β ∈ {1,2,3}:
```
B_{k,α} = ½(∇N_k ⊗ e_α + e_α ⊗ ∇N_k)    [strain-displacement]
K_{kl}[α,β] = B_{k,α} : C : B_{l,β}       [double contraction]
```

# Arguments
- `grad_k`: Physical gradient ∇N_k at integration point
- `grad_l`: Physical gradient ∇N_l at integration point
- `C`: Material stiffness tensor (4th-order symmetric, elasticity or tangent)

# Returns
3×3 stiffness block contribution (before detJ*w scaling)

# Performance
Zero allocations - all tensors stack-allocated.
"""
@inline function compute_block_at_point(
    grad_k::Vec{3,Float64},
    grad_l::Vec{3,Float64},
    C::SymmetricTensor{4,3,Float64}
)
    # Basis vectors (compiler should hoist to caller if in loop)
    e_1 = Vec{3}((1.0, 0.0, 0.0))
    e_2 = Vec{3}((0.0, 1.0, 0.0))
    e_3 = Vec{3}((0.0, 0.0, 1.0))
    e = (e_1, e_2, e_3)

    K_kl_ip = zero(Tensor{2,3,Float64})

    @inbounds for α in 1:3, β in 1:3
        e_α, e_β = e[α], e[β]

        # Strain-displacement B-matrices (symmetric part of ∇u)
        B_k_α = 0.5 * (grad_k ⊗ e_α + e_α ⊗ grad_k)
        B_l_β = 0.5 * (grad_l ⊗ e_β + e_β ⊗ grad_l)

        # Double contraction: σ : ε
        k_αβ = dcontract(B_k_α, dcontract(C, B_l_β))

        K_kl_ip += k_αβ * (e_α ⊗ e_β)
    end

    return K_kl_ip
end

"""
    compute_block!(
        prepared::PreparedElement,
        material::LinearElastic,
        k_local::Int,
        l_local::Int
    ) -> Tensor{2,3}

Compute 3×3 stiffness block between local nodes k and l (fully integrated).

This is the **key interface for nodal assemblers**. Given prepared geometry,
compute coupling between any two nodes without forming full element matrix.

# Algorithm
```
K[k,l] = ∑_q compute_block_at_point(∇N_k^q, ∇N_l^q, C) * detJ_q * w_q
```

# Arguments
- `prepared`: Precomputed element geometry (from `prepare_element!`)
- `material`: Linear elastic material (constant C)
- `k_local`: First local node index (1 to N)
- `l_local`: Second local node index (1 to N)

# Returns
Fully integrated 3×3 stiffness block K[k,l]

# Performance
- Zero allocations (all stack types)
- Reuses prepared gradients (no Jacobian recomputation)
- ~10-20 FLOPs per integration point for linear elastic

# Example (Nodal Assembler Loop)

```julia
for (elem_id, local_i) in elements_touching_node[node_i]
    prepared = prepare_element!(cache, kernel, elem_id, mesh)

    for local_j in 1:nnodes_elem
        block = compute_block!(prepared, material, local_i, local_j)
        node_j = mesh.connectivity[elem_id][local_j]
        accumulate_to_row!(row_buffer, node_j, block)
    end
end
```
"""
@inline function compute_block!(
    prepared::PreparedElement{N,NIP},
    material::LinearElastic,
    k_local::Int,
    l_local::Int
) where {N,NIP}

    # Constant elasticity tensor for linear elastic
    C = elasticity_tensor(material)

    K_kl = zero(Tensor{2,3,Float64})

    # Integrate over all quadrature points
    @inbounds for q in 1:NIP
        grad_k = prepared.∇N_data[q][k_local]
        grad_l = prepared.∇N_data[q][l_local]

        # Atomic block computation at this point
        K_kl_ip = compute_block_at_point(grad_k, grad_l, C)

        # Accumulate with integration weight
        K_kl += K_kl_ip * prepared.detJ_w[q]
    end

    return K_kl
end

"""
    compute_block!(
        prepared::PreparedElement,
        material::NeoHookean,
        k_local::Int,
        l_local::Int,
        u_elem::AbstractVector{Float64}
    ) -> Tensor{2,3}

Compute 3×3 stiffness block for NeoHookean material (strain-dependent tangent).

Requires current displacement `u_elem` to compute deformation gradient F
and tangent modulus 𝔻(E) at each integration point.

# Arguments
- `prepared`: Precomputed element geometry
- `material`: NeoHookean material model
- `k_local`, `l_local`: Local node indices
- `u_elem`: Element displacement DOFs [3N] (for computing F)

# Returns
Fully integrated 3×3 tangent stiffness block
"""
@inline function compute_block!(
    prepared::PreparedElement{N,NIP},
    material::NeoHookean,
    k_local::Int,
    l_local::Int,
    u_elem::AbstractVector{Float64}
) where {N,NIP}

    I = one(Tensor{2,3,Float64})
    K_kl = zero(Tensor{2,3,Float64})

    @inbounds for q in 1:NIP
        ∇N_q = prepared.∇N_data[q]

        # Compute deformation gradient F at this integration point
        F = I
        for k in 1:N
            k_offset = 3(k - 1)
            u_k = Vec{3}((u_elem[k_offset+1], u_elem[k_offset+2], u_elem[k_offset+3]))
            F += u_k ⊗ ∇N_q[k]
        end

        # Right Cauchy-Green and Green-Lagrange strain
        C_tensor = symmetric(F' ⋅ F)
        E = SymmetricTensor{2,3}(0.5 * (C_tensor - I))

        # Material tangent modulus
        _, 𝔻, _ = compute_stress(material, E)

        # Block at this point (using strain-dependent tangent)
        grad_k = ∇N_q[k_local]
        grad_l = ∇N_q[l_local]
        K_kl_ip = compute_block_at_point(grad_k, grad_l, 𝔻)

        K_kl += K_kl_ip * prepared.detJ_w[q]
    end

    return K_kl
end

# ============================================================================
# LEGACY HELPER FUNCTIONS (Kept for backward compatibility, will be deprecated)
# ============================================================================
# These are the old monolithic implementations that compute full Ke matrices.
# New code should use the block API above.
# ============================================================================

"""
    compute_element_stiffness_blocked!(
        K_blocks::Matrix{Tensor{2,3}},
        X::Vector{Vec{3}},
        material::LinearElastic,
        u_elem::Vector{Float64},
        topology::T,
        basis::B,
        ips
    ) -> Nothing

Compute element stiffness for LinearElastic material **in-place**.

Uses constant elasticity tensor C for efficiency.
"""
@inline function compute_element_stiffness_blocked!(
    K_blocks::AbstractMatrix{Tensor{2,3,Float64,9}},
    X::AbstractVector{Vec{3,Float64}},
    material::LinearElastic,
    u_elem::AbstractVector{Float64},
    topology::T,
    basis::B,
    ips
) where {T<:AbstractTopology{N},B<:AbstractBasis} where {N}

    # Pre-compute elasticity tensor once
    C = elasticity_tensor(material)

    # Basis vectors
    e_1, e_2, e_3 = Vec{3}((1.0, 0.0, 0.0)), Vec{3}((0.0, 1.0, 0.0)), Vec{3}((0.0, 0.0, 1.0))
    e = (e_1, e_2, e_3)

    # Integrate over node pairs
    for k in 1:N, l in 1:N
        K_kl = zero(Tensor{2,3,Float64})
        # Accumulate contributions from all integration points
        for ip in ips
            ξ = Vec{3}(ip.ξ)
            w = ip.weight

            # Shape function gradients in reference coordinates
            dN_dξ = get_basis_derivatives(topology, basis, ξ)

            # Jacobian transformation: J = ∑_i X_i ⊗ (∂N_i/∂ξ)
            J = X[1] ⊗ dN_dξ[1]
            for i in 2:N
                J += X[i] ⊗ dN_dξ[i]
            end
            detJ = det(J)
            J_inv = inv(J)
            J_inv_T = transpose(J_inv)

            # Physical gradients
            grad_k = J_inv_T ⋅ dN_dξ[k]
            grad_l = J_inv_T ⋅ dN_dξ[l]

            # Compute stiffness block inline (zero allocations)
            K_kl_ip = zero(Tensor{2,3,Float64})
            for α in 1:3, β in 1:3
                e_α, e_β = e[α], e[β]

                # Strain-displacement B-matrices
                B_k_α = 0.5 * (grad_k ⊗ e_α + e_α ⊗ grad_k)
                B_l_β = 0.5 * (grad_l ⊗ e_β + e_β ⊗ grad_l)

                # Double contraction: B_k : C : B_l
                k_αβ = dcontract(B_k_α, dcontract(C, B_l_β))

                K_kl_ip += k_αβ * (e_α ⊗ e_β)
            end

            # Accumulate with quadrature weight and Jacobian
            K_kl += K_kl_ip * detJ * w
        end
        K_blocks[k, l] = K_kl
    end

    return nothing
end

"""
    compute_element_stiffness_blocked!(
        K_blocks::Matrix{Tensor{2,3}},
        X::Vector{Vec{3}},
        material::NeoHookean,
        u_elem::Vector{Float64},
        topology::T,
        basis::B,
        ips
    ) -> Nothing

Compute element stiffness for NeoHookean material **in-place**.

Uses strain-dependent tangent modulus 𝔻(E).
"""
function compute_element_stiffness_blocked!(
    K_blocks::AbstractMatrix{Tensor{2,3,Float64,9}},
    X::AbstractVector{Vec{3,Float64}},
    material::NeoHookean,
    u_elem::AbstractVector{Float64},
    topology::T,
    basis::B,
    ips
) where {T<:AbstractTopology{N},B<:AbstractBasis} where {N}

    # Basis vectors
    e_1, e_2, e_3 = Vec{3}((1.0, 0.0, 0.0)), Vec{3}((0.0, 1.0, 0.0)), Vec{3}((0.0, 0.0, 1.0))
    e = (e_1, e_2, e_3)

    # Identity tensor
    I = one(Tensor{2,3,Float64})

    # Integrate over integration points
    for ip in ips
        ξ = Vec{3}(ip.ξ)
        w = ip.weight

        # Shape function gradients
        dN_dξ = get_basis_derivatives(topology, basis, ξ)

        # Jacobian
        J = X[1] ⊗ dN_dξ[1]
        for k in 2:N
            J += X[k] ⊗ dN_dξ[k]
        end
        J_inv = inv(J)
        detJ = det(J)
        detJ > 0.0 || error("Negative Jacobian determinant: $detJ")

        # Physical gradients
        ∇N = ntuple(k -> J_inv' ⋅ dN_dξ[k], N)

        # Compute deformation gradient F = I + ∇u
        F = I
        for k in 1:N
            k_offset = 3(k - 1)
            u_k = Vec{3}((u_elem[k_offset+1], u_elem[k_offset+2], u_elem[k_offset+3]))
            F += u_k ⊗ ∇N[k]
        end

        # Right Cauchy-Green tensor C = F^T F
        C_tensor = symmetric(F' ⋅ F)

        # Green-Lagrange strain E = ½(C - I)
        E = SymmetricTensor{2,3}(0.5 * (C_tensor - I))

        # Compute stress and material tangent
        S, 𝔻, _ = compute_stress(material, E)

        # Integration weight
        dV = detJ * w

        # Compute stiffness contributions for each node pair
        for k in 1:N
            grad_k = ∇N[k]
            for l in 1:N
                grad_l = ∇N[l]

                # Accumulate 3×3 block K_kl
                K_kl = zero(Tensor{2,3,Float64})
                for α in 1:3, β in 1:3
                    e_α, e_β = e[α], e[β]

                    # Strain-displacement tensors
                    B_k_α = 0.5 * (grad_k ⊗ e_α + e_α ⊗ grad_k)
                    B_l_β = 0.5 * (grad_l ⊗ e_β + e_β ⊗ grad_l)

                    # Double contraction: B_k^α : 𝔻 : B_l^β
                    value = dcontract(dcontract(B_k_α, 𝔻), B_l_β)

                    # Assemble into K_kl[α,β]
                    K_kl += value * (e_α ⊗ e_β)
                end

                # Accumulate to global block
                K_blocks[k, l] += K_kl * dV
            end
        end
    end

    return nothing
end

"""
    blocked_tensor_to_matrix_view!(K_e::AbstractMatrix, K_blocks::Matrix{Tensor{2,3}})

Convert blocked tensor matrix to Float64 matrix **in-place**.

# Arguments
- `K_e`: Output matrix view [3N × 3N] (modified in-place)
- `K_blocks`: Input blocked matrix [N × N] of Tensor{2,3}

# Performance

Zero allocations - writes directly to output view.
"""
function blocked_tensor_to_matrix_view!(
    K_e::AbstractMatrix{Float64},
    K_blocks::AbstractMatrix{Tensor{2,3,Float64,9}}
)
    N = size(K_blocks, 1)

    @inbounds for k in 1:N, l in 1:N
        K_kl = K_blocks[k, l]
        for α in 1:3, β in 1:3
            i = 3 * (k - 1) + α
            j = 3 * (l - 1) + β
            K_e[i, j] = K_kl[α, β]
        end
    end

    return nothing
end

# ============================================================================
# HELPER FUNCTIONS FOR compute_element_stiffness! (material dispatch)
# ============================================================================
# These enable compile-time dispatch instead of runtime type checks.
# ============================================================================

"""
    compute_all_blocks!(K_blocks, prepared, material::LinearElastic, u_elem, Ncheck)

Helper to compute all blocks for LinearElastic material (doesn't need u_elem).
Uses material dispatch for type stability.
"""
@inline function compute_all_blocks!(
    K_blocks::AbstractMatrix{Tensor{2,3,Float64,9}},
    prepared::PreparedElement{N,NIP},
    material::LinearElastic,
    u_elem::AbstractVector{Float64},
    Ncheck::Int
) where {N,NIP}
    @assert N == Ncheck "Topology mismatch"

    for k in 1:N, l in 1:N
        K_blocks[k, l] = compute_block!(prepared, material, k, l)
    end

    return nothing
end

"""
    compute_all_blocks!(K_blocks, prepared, material::NeoHookean, u_elem, Ncheck)

Helper to compute all blocks for NeoHookean material (requires u_elem for tangent).
Uses material dispatch for type stability.
"""
@inline function compute_all_blocks!(
    K_blocks::AbstractMatrix{Tensor{2,3,Float64,9}},
    prepared::PreparedElement{N,NIP},
    material::NeoHookean,
    u_elem::AbstractVector{Float64},
    Ncheck::Int
) where {N,NIP}
    @assert N == Ncheck "Topology mismatch"

    for k in 1:N, l in 1:N
        K_blocks[k, l] = compute_block!(prepared, material, k, l, u_elem)
    end

    return nothing
end
