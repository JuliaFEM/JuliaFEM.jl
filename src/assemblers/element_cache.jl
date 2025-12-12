# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Element and node cache implementations for zero-allocation assembly.
"""

using Tensors

"""
    ElementCache{T<:AbstractTopology,B<:AbstractBasis,IPS}

Workspace for element-level computations.

Contains pre-allocated arrays for element matrices, vectors, and DOF mapping.
Reused across all elements during assembly (zero allocations!).

# Fields
- `Ke::Matrix{Float64}`: Element stiffness matrix [max_ndofs_elem × max_ndofs_elem]
- `fe::Vector{Float64}`: Element force vector [max_ndofs_elem]
- `K_blocks::Matrix{Tensor{2,3}}`: Blocked stiffness matrix [max_nnodes × max_nnodes]
- `u_buffer::Vector{Vec{3,Float64}}`: Element displacement vectors [max_nnodes]
- `dofs::Vector{Int}`: Global DOF indices [max_ndofs_elem]
- `topology::T`: Pre-computed topology instance
- `basis::B`: Pre-computed basis instance
- `ips::IPS`: Pre-computed integration points

# Note
Geometry-related fields (X, coords, ∇N) removed - now in GeometryCache.
This eliminates duplication and separates concerns.
"""
struct ElementCache{T<:AbstractTopology,B<:AbstractBasis,IPS}
    Ke::Matrix{Float64}                           # Local stiffness matrix (legacy format)
    fe::Vector{Float64}                           # Local force vector (legacy format)
    K_blocks::Matrix{Tensor{2,3,Float64,9}}       # Blocked stiffness matrix [N×N]
    f_blocks::Vector{Vec{3,Float64}}              # Blocked force vector [N]
    u_buffer::Vector{Vec{3,Float64}}              # Element displacement vectors [N]
    dofs::Vector{Int}                             # Global DOF indices
    topology::T                                   # Pre-computed topology
    basis::B                                      # Pre-computed basis
    ips::IPS                                      # Pre-computed integration points
end

"""
    reset!(cache::ElementCache)

Reset element cache to zero values.

# Side Effects
Mutates all arrays in cache to zero.
"""
function reset!(cache::ElementCache)
    fill!(cache.Ke, 0.0)
    fill!(cache.fe, 0.0)
    fill!(cache.K_blocks, zero(Tensor{2,3,Float64,9}))
    fill!(cache.f_blocks, zero(Vec{3,Float64}))
    fill!(cache.u_buffer, zero(Vec{3,Float64}))
    fill!(cache.dofs, 0)
    return nothing
end

# ============================================================================
# CONSTRUCTORS
# ============================================================================

"""
    create_element_cache(mesh::AbstractMesh, kernel::AbstractKernel) -> ElementCache

Create pre-allocated element workspace.

Allocates arrays for element stiffness matrix, force vector, and DOF mapping.
Sizes determined from mesh type parameters and kernel requirements.

# Arguments
- `mesh::Mesh{N,T}`: Mesh with maximum element size N
- `kernel`: Kernel defining DOFs per node

# Returns
- `ElementCache` with pre-allocated buffers sized for largest element

# Pre-computed Data
The cache includes pre-computed topology, basis, and integration points:
- `topology`: Reference element topology (e.g., Tet4())
- `basis`: Lagrange basis functions (e.g., Lagrange{Tet4,1}())
- `ips`: Integration point coordinates and weights

# Example

julia> mesh = Mesh{4,Tet4}(nodes, elements)  # Max 4 nodes per element
julia> kernel = ContinuumKernel()
julia> cache = create_element_cache(mesh, kernel)

# cache.K_blocks is 4x4 matrix of 3x3 blocks (12x12 total)
"""
function create_element_cache(mesh::AbstractMesh, kernel::AbstractKernel)
    # Get maximum element size from Mesh{N,T} type parameters
    MeshType = typeof(mesh)
    max_nnodes_elem = MeshType.parameters[1]::Int
    TopologyType = MeshType.parameters[2]
    ndofs_per_node = dofs_per_node(kernel)
    max_ndofs_elem = max_nnodes_elem * ndofs_per_node

    # Pre-compute topology, basis, and integration points
    topology = TopologyType()
    basis = Lagrange{1}()  # New API: basis order only (topology passed separately)
    ips = integration_points(topology)  # New API: auto-selects quadrature order

    return ElementCache(
        zeros(max_ndofs_elem, max_ndofs_elem),  # Ke (legacy)
        zeros(max_ndofs_elem),                   # fe (legacy)
        Matrix{Tensor{2,3,Float64,9}}(undef, max_nnodes_elem, max_nnodes_elem),  # K_blocks
        [zero(Vec{3,Float64}) for _ in 1:max_nnodes_elem],  # f_blocks
        [zero(Vec{3,Float64}) for _ in 1:max_nnodes_elem],  # u_buffer
        zeros(Int, max_ndofs_elem),              # dofs
        topology,                                # Pre-computed topology
        basis,                                   # Pre-computed basis
        ips                                      # Pre-computed integration points
    )
end

