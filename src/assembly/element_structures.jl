# Traditional Element Assembly
#
# This module provides the standard element-by-element assembly approach
# for comparison with nodal assembly. Builds global tangent stiffness matrix
# and residual force vector using sparse matrix formats.

using Tensors
using SparseArrays
using LinearAlgebra

"""
    ElementAssemblyData{T}

Storage for element assembly using traditional (element-by-element) approach.

# Fields
- `K_global::SparseMatrixCSC{T}`: Global tangent stiffness matrix
- `r_global::Vector{T}`: Global residual force vector (r = f_int - f_ext)
- `f_int_global::Vector{T}`: Global internal force vector
- `f_ext_global::Vector{T}`: Global external force vector
- `ndof::Int`: Total number of degrees of freedom

# Notes
- Assembly uses COO (coordinate) format, then converts to CSC
- Multiple elements can write to same global DOF (summed automatically)
"""
mutable struct ElementAssemblyData{T}
    K_global::SparseMatrixCSC{T,Int}
    r_global::Vector{T}
    f_int_global::Vector{T}
    f_ext_global::Vector{T}
    ndof::Int
end

"""
    ElementAssemblyData(ndof::Int, ::Type{T}=Float64)

Allocate storage for traditional element assembly.

# Arguments
- `ndof`: Total degrees of freedom (nnodes × 3 for 3D)
- `T`: Floating point type (default Float64)

# Example
```julia
nnodes = 100
assembly = ElementAssemblyData(3 * nnodes, Float64)
```
"""
function ElementAssemblyData(ndof::Int, ::Type{T}=Float64) where T
    # Pre-allocate empty sparse matrix (will fill during assembly)
    K_global = spzeros(T, ndof, ndof)
    r_global = zeros(T, ndof)
    f_int_global = zeros(T, ndof)
    f_ext_global = zeros(T, ndof)

    return ElementAssemblyData{T}(K_global, r_global, f_int_global, f_ext_global, ndof)
end

"""
    reset!(assembly::ElementAssemblyData)

Reset assembly data to zero (for incremental/iterative solvers).
"""
function reset!(assembly::ElementAssemblyData{T}) where T
    assembly.K_global = spzeros(T, assembly.ndof, assembly.ndof)
    fill!(assembly.r_global, 0.0)
    fill!(assembly.f_int_global, 0.0)
    fill!(assembly.f_ext_global, 0.0)
end

"""
    ElementContribution{T}

Local element contribution before scattering to global.

# Fields
- `element_id::Int`: Element ID
- `gdofs::Vector{Int}`: Global DOF indices (e.g., [1,2,3,4,5,6,...] for nodes)
- `K_local::Matrix{T}`: Local stiffness matrix (ndofs_local × ndofs_local)
- `f_int_local::Vector{T}`: Local internal force vector
- `f_ext_local::Vector{T}`: Local external force vector

# Notes
- For Tet4: ndofs_local = 12 (4 nodes × 3 DOF)
- For Tet10: ndofs_local = 30 (10 nodes × 3 DOF)
"""
struct ElementContribution{T}
    element_id::Int
    gdofs::Vector{Int}
    K_local::Matrix{T}
    f_int_local::Vector{T}
    f_ext_local::Vector{T}
end

"""
    ElementContribution(element_id::Int, gdofs::Vector{Int}, ::Type{T}=Float64)

Allocate storage for element contribution.

# Arguments
- `element_id`: Element ID
- `gdofs`: Global DOF indices
- `T`: Floating point type

# Example
```julia
# Tet4 element connecting nodes [5, 7, 12, 15]
gdofs = [13,14,15, 19,20,21, 34,35,36, 43,44,45]  # 3 DOF per node
contrib = ElementContribution(1, gdofs, Float64)
```
"""
function ElementContribution(element_id::Int, gdofs::Vector{Int}, ::Type{T}=Float64) where T
    ndofs = length(gdofs)
    K_local = zeros(T, ndofs, ndofs)
    f_int_local = zeros(T, ndofs)
    f_ext_local = zeros(T, ndofs)

    return ElementContribution{T}(element_id, gdofs, K_local, f_int_local, f_ext_local)
end

"""
    scatter_to_global!(assembly::ElementAssemblyData, contrib::ElementContribution)

Scatter element contribution to global matrices/vectors (traditional assembly).

This is the key operation in element assembly: add local element quantities
to global system. Uses COO format (accumulates into lists).

# Arguments
- `assembly`: Global assembly data
- `contrib`: Element contribution

# Notes
- Multiple elements can contribute to same global DOF (summed)
- For GPU: Would require atomic operations (slow!)
- For CPU: Direct scatter-add works fine
"""
function scatter_to_global!(assembly::ElementAssemblyData{T},
    contrib::ElementContribution{T}) where T
    # Scatter forces (simple vector addition)
    for (local_i, global_i) in enumerate(contrib.gdofs)
        assembly.f_int_global[global_i] += contrib.f_int_local[local_i]
        assembly.f_ext_global[global_i] += contrib.f_ext_local[local_i]
    end

    # Scatter stiffness (matrix addition)
    # Build list of (I, J, V) triplets for sparse matrix
    I_rows = Int[]
    J_cols = Int[]
    values = T[]

    ndofs_local = length(contrib.gdofs)
    for i in 1:ndofs_local, j in 1:ndofs_local
        if abs(contrib.K_local[i, j]) > 1e-14  # Skip near-zeros
            push!(I_rows, contrib.gdofs[i])
            push!(J_cols, contrib.gdofs[j])
            push!(values, contrib.K_local[i, j])
        end
    end

    # Add to existing sparse matrix
    K_elem = sparse(I_rows, J_cols, values, assembly.ndof, assembly.ndof)
    assembly.K_global += K_elem
end

"""
    compute_residual!(assembly::ElementAssemblyData)

Compute residual force vector: r = f_int - f_ext

Should be called after all elements have been assembled.
"""
function compute_residual!(assembly::ElementAssemblyData{T}) where T
    assembly.r_global .= assembly.f_int_global .- assembly.f_ext_global
end

"""
    assemble_elements!(assembly::ElementAssemblyData,
                      contributions::Vector{ElementContribution})

Assemble all element contributions to global system.

# Arguments
- `assembly`: Global assembly data (modified in-place)
- `contributions`: Vector of element contributions

# Example
```julia
assembly = ElementAssemblyData(ndof)
contributions = compute_all_element_contributions(elements, u, time)
assemble_elements!(assembly, contributions)
compute_residual!(assembly)

# Now solve: K_global * Δu = -r_global
```
"""
function assemble_elements!(assembly::ElementAssemblyData{T},
    contributions::Vector{ElementContribution{T}}) where T
    reset!(assembly)

    # Loop over elements and scatter (element assembly)
    for contrib in contributions
        scatter_to_global!(assembly, contrib)
    end

    # Compute residual
    compute_residual!(assembly)
end

"""
    apply_dirichlet_bc!(assembly::ElementAssemblyData, 
                       fixed_dofs::Vector{Int},
                       prescribed_values::Vector{T}=zeros(length(fixed_dofs)))

Apply Dirichlet (essential) boundary conditions by penalty method.

# Arguments
- `assembly`: Global assembly data (modified in-place)
- `fixed_dofs`: DOF indices to fix
- `prescribed_values`: Prescribed displacement values (default: zeros)

# Method
Uses penalty method: adds large stiffness to diagonal and corresponding RHS.

For DOF i with prescribed value u_prescribed:
- K[i,i] += penalty (e.g., 1e10 * max_K)
- r[i] = penalty * (u_current - u_prescribed)

# Example
```julia
# Fix nodes 1 and 2 in all directions (zero displacement)
fixed_dofs = [1,2,3, 4,5,6]  # Nodes 1,2 × 3 DOF
apply_dirichlet_bc!(assembly, fixed_dofs)
```
"""
function apply_dirichlet_bc!(assembly::ElementAssemblyData{T},
    fixed_dofs::Vector{Int},
    prescribed_values::Vector{T}=zeros(T, length(fixed_dofs))) where T
    # Penalty parameter (large relative to stiffness)
    max_K = maximum(abs, assembly.K_global)
    penalty = 1e10 * max_K

    for (idx, dof) in enumerate(fixed_dofs)
        # Add penalty stiffness to diagonal
        assembly.K_global[dof, dof] += penalty

        # Modify residual (assuming current displacement is zero for now)
        # In full Newton: r[i] += penalty * (u_current[i] - u_prescribed[i])
        assembly.r_global[dof] = penalty * prescribed_values[idx]
    end
end

"""
    get_dof_indices(connectivity::NTuple{N,Int}, dim::Int=3) -> Vector{Int}

Get global DOF indices for an element given node connectivity.

# Arguments
- `connectivity`: Element node IDs (e.g., (5, 7, 12, 15) for Tet4)
- `dim`: Dimension (3 for 3D elasticity)

# Returns
- `gdofs::Vector{Int}`: Global DOF indices

# Example
```julia
# Element with nodes [5, 7, 12, 15]
gdofs = get_dof_indices((5, 7, 12, 15), 3)
# Returns: [13,14,15, 19,20,21, 34,35,36, 43,44,45]
```
"""
function get_dof_indices(connectivity::NTuple{N,Int}, dim::Int=3) where N
    nnodes = length(connectivity)
    gdofs = zeros(Int, dim * nnodes)

    for (local_i, global_node) in enumerate(connectivity)
        for d in 1:dim
            gdofs[dim*(local_i-1)+d] = dim * (global_node - 1) + d
        end
    end

    return gdofs
end

"""
    matrix_vector_product(assembly::ElementAssemblyData, v::Vector{T}) -> Vector{T}

Compute matrix-vector product: w = K * v using assembled sparse matrix.

# Arguments
- `assembly`: Assembly data (contains K_global)
- `v`: Input vector (ndof)

# Returns
- `w`: Output vector w = K * v

# Example
```julia
# GMRES matrix-free operator
function matvec(v)
    return matrix_vector_product(assembly, v)
end
Δu = gmres(matvec, -r, tol=1e-6)
```
"""
function matrix_vector_product(assembly::ElementAssemblyData{T}, v::Vector{T}) where T
    return assembly.K_global * v
end

"""
    print_assembly_stats(assembly::ElementAssemblyData)

Print statistics about assembled system (for debugging).
"""
function print_assembly_stats(assembly::ElementAssemblyData)
    nnz_K = nnz(assembly.K_global)
    ndof = assembly.ndof
    fill_ratio = nnz_K / (ndof * ndof)

    println("="^60)
    println("Traditional Element Assembly Statistics")
    println("="^60)
    println("  Total DOF:              ", ndof)
    println("  K matrix size:          ", size(assembly.K_global))
    println("  K non-zeros:            ", nnz_K)
    println("  K fill ratio:           ", round(fill_ratio, sigdigits=2))
    println("  K memory (MB):          ", round(nnz_K * 16 / 1024^2, digits=2))
    println("  ||f_int||:              ", round(norm(assembly.f_int_global), sigdigits=2))
    println("  ||f_ext||:              ", round(norm(assembly.f_ext_global), sigdigits=2))
    println("  ||residual||:           ", round(norm(assembly.r_global), sigdigits=2))
    println("  K symmetric:            ", issymmetric(assembly.K_global))
    println("="^60)
end

# Export main types and functions
export ElementAssemblyData, ElementContribution
export reset!, scatter_to_global!, compute_residual!
export assemble_elements!, apply_dirichlet_bc!
export get_dof_indices, matrix_vector_product
export print_assembly_stats
