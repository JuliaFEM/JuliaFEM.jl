# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Element and node cache implementations for zero-allocation assembly.
"""

using Tensors

"""
    ContinuumElementCache{T<:AbstractTopology,B<:AbstractBasis,IPS}

Per-element workspace used by the element-based COO assembler with
3D continuum-mechanics kernels (`ContinuumKernel` and friends).

Contains pre-allocated arrays for blocked element matrices, displacement
buffers, DOF mapping, and the element's compile-time topology / basis /
integration-point bundle. Reused across all elements during assembly,
so the inner loop is allocation-free.

The block layout (`Tensor{2,3,Float64,9}` and `Vec{3,Float64}`) is
hard-coded for 3D continuum mechanics. Heat, thermo-elastic, and the
mixed u–p / Hellinger–Reissner / Hu–Washizu kernels bypass this struct
and carry their own per-element scratch on the DOF-based assembler.

The shorter alias `ElementCache` is kept for backwards compatibility.

# Fields
- `K_blocks::Matrix{Tensor{2,3,Float64,9}}` — blocked stiffness `[N × N]`
- `f_blocks::Vector{Vec{3,Float64}}` — blocked force `[N]`
- `u_buffer::Vector{Vec{3,Float64}}` — element displacement `[N]`
- `dofs::Vector{Int}` — global DOF indices `[max_ndofs_elem]`
- `topology::T` — pre-computed topology instance
- `basis::B` — pre-computed basis instance
- `ips::IPS` — pre-computed integration points
"""
struct ContinuumElementCache{T<:AbstractTopology,B<:AbstractBasis,IPS}
    K_blocks::Matrix{Tensor{2,3,Float64,9}}
    f_blocks::Vector{Vec{3,Float64}}
    u_buffer::Vector{Vec{3,Float64}}
    dofs::Vector{Int}
    topology::T
    basis::B
    ips::IPS
end

"""
    ElementCache

Backwards-compatible alias for [`ContinuumElementCache`](@ref). New code
should prefer the explicit name to make the continuum-only assumption
visible.
"""
const ElementCache = ContinuumElementCache

"""
    reset!(cache::ContinuumElementCache)

Zero out all storage so the cache can be reused for the next element.
"""
function reset!(cache::ContinuumElementCache)
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
    create_element_cache(mesh::AbstractMesh, kernel::AbstractKernel) -> ContinuumElementCache

Create a pre-allocated continuum element workspace.

Allocates arrays for element stiffness matrix, force vector, and DOF
mapping. Sizes are derived from the mesh type parameters and the
kernel's `dofs_per_node`.

The `dofs::Vector{Int}` slot must hold every entry of `elem.dof_indices`
from [`create_elements!`](@ref). When unknowns live on edges or faces,
that length can exceed `nnodes_per_elem * dofs_per_node(kernel)` (e.g.
[`FacetMassKernel`](@ref) / [`EdgeMassKernel`](@ref) use `dofs_per_node = 1`
but 6 or 12 local facet DOFs on Hex8). Pass `max_local_dofs` so the
buffer is large enough; it defaults to `0`, meaning only the nodal
product is used.

# Arguments
- `mesh::Mesh{N,T}` — mesh with up to `N` nodes per element.
- `kernel` — kernel defining DOFs per node.
- `max_local_dofs` — if positive, `length(dofs)` is at least this value.
- `basis` — optional [`AbstractBasis`](@ref) instance; defaults to [`Lagrange{1}`](@ref).
  [`DOFBasedCOOCache`](@ref) passes [`basis_type`](@ref) from the first element so
  quadratic meshes (`Tet10`, `Hex20`, …) match [`get_basis_functions`](@ref).
  Quadrature uses [`integration_points(topology, basis)`](@ref) so the rule tracks
  basis order, not just topology node count.

# Returns
A `ContinuumElementCache` whose buffers are sized for the largest
element in the mesh, plus the pre-computed topology, basis, and
integration points.
"""
function create_element_cache(
    mesh::AbstractMesh,
    kernel::AbstractKernel;
    max_local_dofs::Int = 0,
    basis = nothing,
)
    MeshType = typeof(mesh)
    max_nnodes_elem = MeshType.parameters[1]::Int
    TopologyType = MeshType.parameters[2]
    ndofs_per_node = dofs_per_node(kernel)
    base_ndofs = max_nnodes_elem * ndofs_per_node
    max_ndofs_elem = max_local_dofs > 0 ? max(base_ndofs, max_local_dofs) : base_ndofs

    topology = TopologyType()
    basis_inst = basis === nothing ? Lagrange{1}() : basis
    ips = integration_points(topology, basis_inst)

    return ContinuumElementCache(
        Matrix{Tensor{2,3,Float64,9}}(undef, max_nnodes_elem, max_nnodes_elem),
        [zero(Vec{3,Float64}) for _ in 1:max_nnodes_elem],
        [zero(Vec{3,Float64}) for _ in 1:max_nnodes_elem],
        zeros(Int, max_ndofs_elem),
        topology,
        basis_inst,
        ips,
    )
end

