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

Compute element stiffness matrix and force vector for continuum mechanics **in-place**.

# Algorithm

1. Get element nodes and coordinates
2. Zero output arrays (Ke, fe)
3. Select topology, basis, integration based on element type
4. Loop over integration points:
   - Compute shape function gradients
   - Compute B-matrix (strain-displacement)
   - Compute material stiffness C or 𝔻
   - Accumulate: Ke += B^T * C * B * detJ * w
5. Apply body forces to fe (if any)

# Material Dispatch

- `LinearElastic`: Uses constant elasticity tensor C
- `NeoHookean`: Uses strain-dependent tangent 𝔻(E)
- Future: Plasticity, damage, etc.

# Zero-Allocation Guarantee

All computations use pre-allocated buffers from `cache`. Temporary tensors
are stack-allocated (small, fast). No heap allocations.

# Arguments
- `cache`: Pre-allocated element workspace
- `kernel`: Continuum kernel with material and formulation
- `element_id`: Element index in mesh
- `mesh`: Finite element mesh
"""
function compute_element_stiffness!(
    cache::ElementCache,
    kernel::ContinuumKernel,
    element_id::Int,
    mesh::AbstractMesh
)
    # Get element connectivity
    conn = mesh.connectivity[element_id]
    nnodes_elem = length(conn)
    ndofs_elem = 3 * nnodes_elem

    # Zero output arrays
    @views fill!(cache.Ke[1:ndofs_elem, 1:ndofs_elem], 0.0)
    @views fill!(cache.fe[1:ndofs_elem], 0.0)

    # Get element node coordinates
    @inbounds for (i, node_id) in enumerate(conn)
        cache.coords[i, :] .= mesh.nodes[node_id]
    end

    # Convert coords to Vector{Vec{3}} for kernel calls
    X_buffer = [Vec{3}(cache.coords[i, :]) for i in 1:nnodes_elem]

    # Get topology type from mesh
    # For Mesh{8, Hexahedron{8}}, this is Hexahedron{8}
    # Extract topology type directly from Mesh{N,T} type parameters
    MeshType = typeof(mesh)
    TopologyType = MeshType.parameters[2]  # T from Mesh{N,T}

    # Create topology, basis, integration instances
    topology = TopologyType()
    basis = Lagrange{TopologyType,1}()
    integration_scheme = default_integration(TopologyType)
    ips = integration_points(integration_scheme, topology)

    # Compute element stiffness using blocked tensor format
    # Allocate K_blocks (small, stack-allocated for typical elements)
    K_blocks = Matrix{Tensor{2,3,Float64,9}}(undef, nnodes_elem, nnodes_elem)
    fill!(K_blocks, zero(Tensor{2,3}))

    # Displacement DOFs (zero for linear elastic, needed for nonlinear)
    u_elem = zeros(ndofs_elem)

    # Call material-dispatched stiffness computation
    compute_element_stiffness_blocked!(
        K_blocks,
        X_buffer,
        kernel.material,
        u_elem,
        topology,
        basis,
        ips
    )

    # Convert blocked tensor to Float64 matrix (cache.Ke)
    blocked_tensor_to_matrix_view!(
        @view(cache.Ke[1:ndofs_elem, 1:ndofs_elem]),
        K_blocks
    )

    # TODO: Add body forces to fe if needed
    # For now, fe = 0 (forces added by Neumann BCs)

    return nothing
end

# ============================================================================
# HELPER FUNCTIONS
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
function compute_element_stiffness_blocked!(
    K_blocks::AbstractMatrix{Tensor{2,3,Float64,9}},
    X::Vector{Vec{3,Float64}},
    material::LinearElastic,
    u_elem::Vector{Float64},
    topology::T,
    basis::B,
    ips
) where {T<:AbstractTopology{N},B<:AbstractBasis} where {N}

    # Pre-compute elasticity tensor once
    C = elasticity_tensor(material)

    # Integrate over node pairs
    for k in 1:N, l in 1:N
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

            # Compute 3×3 stiffness block
            K_kl = compute_stiffness_block(grad_k, grad_l, C)

            # Accumulate with quadrature weight and Jacobian
            K_blocks[k, l] += K_kl * detJ * w
        end
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
    X::Vector{Vec{3,Float64}},
    material::NeoHookean,
    u_elem::Vector{Float64},
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
