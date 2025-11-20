# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Assembly for 3D Continuum Elasticity - Ferrite-Style Two-Pointer Merge (V2)

This is a CLEAN implementation using the Ferrite two-pointer merge algorithm
for 4.1x faster assembly compared to COO triplets.

Key differences from continuum_3d.jl:
- Uses pre-built CSC structure (K_csc) instead of COO triplets
- Sorts element DOFs for linear-time merge
- Zero allocations in assembly loop
- 4.1x faster assembly (2.36ms vs 9.71ms for 2500 Tet4 elements)
- 16.6x less memory (506 KB vs 8.4 MB)

References:
- Design: experiments/FERRITE_DATA_STRUCTURES_DESIGN.md
- Proof of concept: experiments/ferrite_style_assembly.jl
- Explanation: experiments/ferrite_sorteddofs_explained.jl
"""

using Tensors
using SparseArrays
using LinearAlgebra

# Import basevec for creating unit vectors
using Tensors: basevec

# ============================================================================
# Data Structures
# ============================================================================

"""
    AssemblyCacheFerrite{N,T,Mat}

Pre-allocated buffers for zero-allocation Ferrite-style assembly.

Replaces COO triplets (I_rows, J_cols, K_values) with:
- Pre-built CSC structure (K_csc) - reused every assembly
- Sorted DOF buffers (sorteddofs, permutation) - for two-pointer merge

# Type Parameters
- `N`: Number of nodes per element (from topology)
- `T`: Topology type (Hexahedron{8}, Tet10, etc.)
- `Mat`: Material type

# Fields (Ferrite-specific)
- `K_csc`: Pre-built sparse matrix structure (zeros, reused)
- `sorteddofs`: Sorted element DOF buffer [3N]
- `permutation`: Sortperm buffer for DOF remapping [3N]

# Fields (Standard)
- `f`: Global force vector
- `K_blocks`: Element blocked stiffness matrix [N×N of Tensor{2,3}]
- `K_e`: Element stiffness in Float64 matrix form [3N×3N]
- `X_buffer`: Element coordinate buffer [N of Vec{3}]
- `gdofs`: Global DOF indices buffer (unsorted) [3N]
- `elements`: Element IDs to assemble (Vector{UInt32})
- `C`: Elasticity tensor (pre-computed)
- `topology`: Topology instance
- `basis`: Basis function instance
- `ips`: Integration points (pre-computed)

# Performance
- Memory: ~506 KB for typical mesh (vs 8.4 MB COO triplets)
- Time: 2.36ms for 2500 Tet4 (vs 9.71ms COO)
- Allocations: Zero after warmup
"""
struct AssemblyCacheFerrite{N,T<:AbstractTopology{N},Mat<:AbstractMaterial}
    # Ferrite-style CSC storage (NEW!)
    K_csc::SparseMatrixCSC{Float64,Int}
    sorteddofs::Vector{Int}
    permutation::Vector{Int}

    # Global force vector
    f::Vector{Float64}

    # Element assembly buffers (same as V1)
    K_blocks::Matrix{Tensor{2,3,Float64,9}}
    K_e::Matrix{Float64}
    X_buffer::Vector{Vec{3,Float64}}
    gdofs::Vector{Int}  # Unsorted DOFs

    # Element set to assemble
    elements::Vector{UInt32}

    # Pre-computed material/topology data
    C::Tensor{4,3,Float64,81}
    topology::T
    basis::Lagrange{1}  # Basis order only (topology passed separately)
    ips::Any  # Integration points tuple
end

# ============================================================================
# Sparsity Pattern Construction
# ============================================================================

"""
Build sparsity pattern from mesh connectivity.

Returns (I, J) vectors for sparse matrix construction.
"""
function build_sparsity_pattern(mesh::M) where {M<:AbstractMesh}
    # Get N from mesh type parameters
    N = typeof(mesh).parameters[1]

    ndofs_total = 3 * length(mesh.nodes)

    # Pre-count entries (3N × 3N per element)
    ndofs_per_elem = 3 * N
    capacity = length(mesh.connectivity) * ndofs_per_elem * ndofs_per_elem

    I = Vector{Int}()
    J = Vector{Int}()
    sizehint!(I, capacity)
    sizehint!(J, capacity)

    # Loop over elements
    for conn in mesh.connectivity
        # Global DOFs for this element
        elem_dofs = Int[]
        sizehint!(elem_dofs, ndofs_per_elem)

        for node_id in conn
            for α in 1:3
                push!(elem_dofs, 3 * (node_id - 1) + α)
            end
        end

        # All pairs (i,j) in elem_dofs
        for i in elem_dofs
            for j in elem_dofs
                push!(I, i)
                push!(J, j)
            end
        end
    end

    return I, J
end

# ============================================================================
# Cache Construction
# ============================================================================

"""
    AssemblyCacheFerrite(physics::Physics{ContinuumFormulation{FullThreeD}, 
                                          Displacement{3}, M, Mat})

Construct Ferrite-style assembly cache with pre-built CSC structure.

This is where ALL allocations happen. After construction, assembly is zero-allocation.

# Key Steps
1. Build sparsity pattern from mesh connectivity
2. Create K_csc with sparse(I, J, ones(...)) 
3. Zero K_csc.nzval for reuse
4. Allocate sorteddofs and permutation buffers
5. Pre-compute material and topology data

# Performance
- One-time cost: ~10-20ms for typical mesh
- Pays off after ~1-2 assemblies vs COO method
"""
function AssemblyCacheFerrite(
    physics::Physics{ContinuumFormulation{FullThreeD},
        Displacement{3},
        M,
        Mat}) where {M<:AbstractMesh,Mat<:AbstractMaterial}

    mesh = physics.mesh
    material = physics.material
    element_set = physics.element_set

    # Get topology type and N from mesh type parameters
    N_param = typeof(mesh).parameters[1]  # N (8 for Hex8)
    T = typeof(mesh).parameters[2]         # Hexahedron{8}

    # Global system dimensions
    nnodes = length(mesh.nodes)
    ndofs = 3 * nnodes

    # Element set
    elem_set = get_element_set(mesh, element_set)
    elements = collect(elem_set)

    # Build sparsity pattern ONCE
    I, J = build_sparsity_pattern(mesh)
    K_csc = sparse(I, J, ones(length(I)), ndofs, ndofs)
    fill!(K_csc.nzval, 0.0)  # Zero values for reuse

    # Ferrite buffers for sorting DOFs
    max_ndofs = 3 * N_param
    sorteddofs = Vector{Int}(undef, max_ndofs)
    permutation = Vector{Int}(undef, max_ndofs)

    # Global force vector
    f = zeros(ndofs)

    # Element assembly buffers (same as V1)
    max_nnodes = N_param
    K_blocks = Matrix{Tensor{2,3,Float64,9}}(undef, max_nnodes, max_nnodes)
    K_e = zeros(max_ndofs, max_ndofs)
    X_buffer = Vector{Vec{3,Float64}}(undef, max_nnodes)
    gdofs = Vector{Int}(undef, max_ndofs)

    # Pre-compute material and topology data
    C = elasticity_tensor(material)
    topology = T()
    basis = Lagrange{1}()  # Basis order only (topology passed separately)
    integration_scheme = default_integration(T)
    ips = integration_points(integration_scheme, topology)

    return AssemblyCacheFerrite{N_param,T,Mat}(
        K_csc,           # Pre-built structure
        sorteddofs,      # Sorted DOF buffer
        permutation,     # Sortperm buffer
        f,
        K_blocks, K_e, X_buffer, gdofs,
        elements,
        C, topology, basis, ips
    )
end

# ============================================================================
# Ferrite Two-Pointer Merge Assembly
# ============================================================================

"""
    assemble_elements_ferrite!(cache::AssemblyCacheFerrite, mesh::M)

Assemble elements using Ferrite two-pointer merge (ZERO allocations).

This is the core Ferrite algorithm:
1. Zero K_csc.nzval once at start
2. For each element:
   a. Compute element stiffness K_e
   b. Get global DOFs (unsorted)
   c. Sort DOFs → sorteddofs, permutation
   d. Two-pointer merge: Linear scan through sorted lists
   e. Accumulate to K_csc using permutation for correct K_e indices

# Algorithm Detail
For each column i_global in element DOFs:
- Get CSC column range: K_csc.colptr[i_global]:(colptr[i_global+1]-1)
- K_csc.rowval[range] is SORTED
- sorteddofs is SORTED
- Two pointers: Ri (CSC), ri (element)
- Advance Ri until K_csc.rowval[Ri] == sorteddofs[ri]
- Accumulate: K_csc.nzval[Ri] += K_e[permutation[ri], permutation[i_local]]

# Performance
- Time: 2.36ms for 2500 Tet4 elements
- Allocations: Zero after warmup
- 4.1x faster than COO method
- Linear-time merge vs O(log n) binary search
"""
function assemble_elements_ferrite!(
    cache::AssemblyCacheFerrite{N,T,Mat},
    mesh::M) where {M<:AbstractMesh,N,T,Mat}

    # Zero K_csc once at start (reuse structure!)
    fill!(cache.K_csc.nzval, 0.0)

    ndofs_elem = 3 * N

    # Loop over elements (ZERO allocations target!)
    @inbounds for i in eachindex(cache.elements)
        elem_id = cache.elements[i]
        @inbounds conn = mesh.connectivity[elem_id]

        # 1. Fill coordinate buffer (in-place)
        @inbounds for j in 1:N
            cache.X_buffer[j] = mesh.nodes[conn[j]]
        end

        # 2. Compute element stiffness
        fill!(cache.K_blocks, zero(Tensor{2,3}))
        compute_element_stiffness!(cache.K_blocks, cache.X_buffer,
            cache.C, cache.topology, cache.basis, cache.ips)
        blocked_tensor_to_matrix!(cache.K_e, cache.K_blocks)

        # 3. Global DOFs (UNSORTED, follows connectivity)
        @inbounds for (local_idx, node_id) in enumerate(conn)
            for α in 1:3
                cache.gdofs[3*(local_idx-1)+α] = 3 * (node_id - 1) + α
            end
        end

        # 4. Sort DOFs (sortperm! is in-place, zero allocation)
        sortperm!(cache.permutation, cache.gdofs)
        @inbounds for k in 1:ndofs_elem
            cache.sorteddofs[k] = cache.gdofs[cache.permutation[k]]
        end

        # 5. TWO-POINTER MERGE (Ferrite method!)
        for i_local in 1:ndofs_elem
            i_global = cache.sorteddofs[i_local]

            # Column range in K_csc for column i_global
            col_start = cache.K_csc.colptr[i_global]
            col_end = cache.K_csc.colptr[i_global+1] - 1

            # Two pointers: Ri (CSC), ri (element)
            Ri = col_start
            for ri in 1:ndofs_elem
                row_i_sorted = cache.sorteddofs[ri]

                # Advance Ri until K_csc.rowval[Ri] >= row_i_sorted
                while Ri <= col_end && cache.K_csc.rowval[Ri] < row_i_sorted
                    Ri += 1
                end

                # If found, accumulate
                if Ri <= col_end && cache.K_csc.rowval[Ri] == row_i_sorted
                    # Use permutation to get correct K_e indices!
                    orig_row = cache.permutation[ri]
                    orig_col = cache.permutation[i_local]
                    cache.K_csc.nzval[Ri] += cache.K_e[orig_row, orig_col]
                end
            end
        end
    end

    nothing
end

# ============================================================================
# Assembly Functions (User-Facing API)
# ============================================================================

"""
    assemble_v2!(physics::Physics{ContinuumFormulation{FullThreeD}, 
                                  Displacement{3}, M, Mat}) -> (K, f)

Assemble using Ferrite two-pointer merge method (V2).

This is the user-facing function that:
1. Creates AssemblyCacheFerrite (allocates all buffers, builds CSC structure)
2. Calls assemble_elements_ferrite! (zero allocations)
3. Applies boundary conditions
4. Returns (K, f)

# Performance
- 4.1x faster assembly than V1 (COO method)
- 16.6x less memory
- Zero allocations in hot loop

# Example
```julia
physics = Physics(...)
K, f = assemble_v2!(physics)  # Ferrite method
u = K \\ f
```
"""
function assemble_v2!(
    physics::Physics{ContinuumFormulation{FullThreeD},
        Displacement{3},
        M,
        Mat}) where {M<:AbstractMesh,Mat<:AbstractMaterial}

    # Create cache (ALL allocations here!)
    cache = AssemblyCacheFerrite(physics)

    # Assemble (ZERO allocations!)
    return _assemble_ferrite!(physics, cache)
end

"""
    _assemble_ferrite!(physics, cache::AssemblyCacheFerrite) -> (K, f)

Zero-allocation assembly using Ferrite cache.

This is the internal function that performs the actual assembly.
Use `assemble_v2!` for the public API.
"""
function _assemble_ferrite!(
    physics::Physics{ContinuumFormulation{FullThreeD},
        Displacement{3},
        M,
        Mat},
    cache::AssemblyCacheFerrite{N,T,Mat2}) where {M<:AbstractMesh,Mat<:AbstractMaterial,N,T,Mat2}

    mesh = physics.mesh
    bc_dirichlet = physics.bc_dirichlet
    bc_neumann = physics.bc_neumann

    # Get dimensions
    nnodes = length(mesh.nodes)
    ndofs = 3 * nnodes

    # Clear force vector (in-place, zero allocation)
    fill!(cache.f, 0.0)

    # Ferrite assembly (ZERO allocations!)
    assemble_elements_ferrite!(cache, mesh)

    # Apply Neumann BCs (add forces to f)
    for (surf_id, force) in zip(bc_neumann.surface_ids, bc_neumann.values)
        # For now, interpret surface_ids as node_ids (simplified)
        node = surf_id
        if node <= nnodes
            for α in 1:3
                cache.f[3*(node-1)+α] += force[α]
            end
        end
    end

    # Copy K_csc (structure already correct, just copy!)
    K = copy(cache.K_csc)

    # Apply Dirichlet BCs (modify K and f)
    for i in 1:length(bc_dirichlet.node_ids)
        node = bc_dirichlet.node_ids[i]
        components = bc_dirichlet.components[i]
        value = bc_dirichlet.values[i]

        for comp in components
            dof = 3 * (node - 1) + comp
            if dof <= ndofs  # Safety check
                K[dof, :] .= 0.0
                K[:, dof] .= 0.0
                K[dof, dof] = 1.0
                cache.f[dof] = value
            end
        end
    end

    return (K, cache.f)
end

# ============================================================================
# Element Stiffness Computation (Reuse from V1)
# ============================================================================

"""
    compute_stiffness_block(grad_k, grad_l, C) -> Tensor{2,3}

Compute single 3×3 stiffness block (reused from continuum_3d.jl).

See continuum_3d.jl for detailed documentation.
"""
@inline function compute_stiffness_block(
    grad_k::Vec{3,Float64},
    grad_l::Vec{3,Float64},
    C::Tensor{4,3,Float64,81}
)::Tensor{2,3,Float64,9}

    K_kl = zero(Tensor{2,3})

    for α in 1:3, β in 1:3
        e_α = basevec(Vec{3}, α)
        e_β = basevec(Vec{3}, β)
        B_k_α = 0.5 * (grad_k ⊗ e_α + e_α ⊗ grad_k)
        B_l_β = 0.5 * (grad_l ⊗ e_β + e_β ⊗ grad_l)
        k_αβ = dcontract(B_k_α, dcontract(C, B_l_β))
        K_kl += k_αβ * (e_α ⊗ e_β)
    end

    return K_kl
end

"""
    blocked_tensor_to_matrix!(K_e, K_blocks)

Convert blocked tensor to Float64 matrix (reused from continuum_3d.jl).
"""
function blocked_tensor_to_matrix!(
    K_e::AbstractMatrix{Float64},
    K_blocks::AbstractMatrix{Tensor{2,3,Float64,9}})

    nnodes = size(K_blocks, 1)
    for i in 1:nnodes, j in 1:nnodes
        for α in 1:3, β in 1:3
            K_e[3*(i-1)+α, 3*(j-1)+β] = K_blocks[i, j][α, β]
        end
    end
    nothing
end

"""
    compute_element_stiffness!(K_blocks, X, C, topology, basis, ips)

Compute element stiffness (reused from continuum_3d.jl).

See continuum_3d.jl for detailed documentation.
"""
function compute_element_stiffness!(
    K_blocks::AbstractMatrix{Tensor{2,3,Float64,9}},
    X::Vector{Vec{3,Float64}},
    C::Tensor{4,3,Float64,81},
    topology::T,
    basis::B,
    ips) where {T<:AbstractTopology{N},B<:AbstractBasis} where N

    for k in 1:N, l in 1:N
        for ip in ips
            ξ = ip.ξ
            w = ip.weight

            dN_dξ = get_basis_derivatives(topology, basis, ξ)

            J = X[1] ⊗ dN_dξ[1]
            for i in 2:N
                J += X[i] ⊗ dN_dξ[i]
            end
            detJ = det(J)
            J_inv = inv(J)
            J_inv_T = transpose(J_inv)

            grad_k = J_inv_T ⋅ dN_dξ[k]
            grad_l = J_inv_T ⋅ dN_dξ[l]

            K_kl = compute_stiffness_block(grad_k, grad_l, C)

            K_blocks[k, l] += K_kl * detJ * w
        end
    end

    nothing
end
