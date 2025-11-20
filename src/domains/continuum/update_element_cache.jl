# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Element cache update functions for continuum elements.

Extracts element displacements and DOF mapping from global data.
"""

"""
    update_element_cache!(
        element_cache::ElementCache,
        kernel::AbstractKernel,
        elem_id::Int,
        mesh::AbstractMesh,
        u_global::Union{Nothing,Vector{Vec{3,Float64}}} = nothing
    )

Update element cache for current element.

Extracts:
- Element displacements → element_cache.u_buffer (if u_global provided)
- DOF mapping → element_cache.dofs

# Arguments
- `element_cache`: Element cache to update
- `kernel`: Domain kernel (provides dofs_per_node)
- `elem_id`: Current element ID
- `mesh`: Finite element mesh
- `u_global`: Global displacement field [nnodes] as Vec{3} (nothing for linear analysis)

# Side Effects
Mutates element_cache.u_buffer, element_cache.dofs

# Zero-Allocation Guarantee
No allocations - writes to pre-allocated element_cache.u_buffer and element_cache.dofs vectors.

# Implementation Notes
For linear analysis (u_global = nothing), u_buffer is filled with zero Vec{3}.
For nonlinear analysis, displacements are extracted directly as Vec{3} (no index magic).
"""
function update_element_cache!(
    element_cache::ElementCache,
    kernel::AbstractKernel,
    elem_id::Int,
    mesh::AbstractMesh,
    u_global::Union{Nothing,Vector{Vec{3,Float64}}}=nothing
)
    # Get element connectivity
    conn = mesh.connectivity[elem_id]
    nnodes_elem = length(conn)

    # Extract element displacements
    if u_global !== nothing
        for (i, node) in enumerate(conn)
            element_cache.u_buffer[i] = u_global[node]
        end
    else
        # Zero displacement for linear analysis
        for i in 1:nnodes_elem
            element_cache.u_buffer[i] = zero(Vec{3,Float64})
        end
    end

    # Get DOF mapping
    ndofs_per_node = dofs_per_node(kernel)
    ndofs_elem = nnodes_elem * ndofs_per_node
    dofs = @view element_cache.dofs[1:ndofs_elem]
    get_dof_mapping!(dofs, kernel, elem_id, mesh)

    return nothing
end
