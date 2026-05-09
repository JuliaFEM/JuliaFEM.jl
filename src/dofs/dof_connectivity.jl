# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
DOF connectivity builders for DOF-based assembly.

Provides inverse mapping: DOF → Elements (which elements touch each DOF).

GPU-Compatible: Uses bits types defined in `dof_connectivity_types.jl`.
Zero-Allocation: After initialization, all operations are allocation-free.
"""

# Import element and DOF handler interfaces (defined in elements.jl and dof_handler.jl)
using ..JuliaFEM: element_dofs, DOFHandler

# ============================================================================
# DOF CONNECTIVITY BUILDERS
# ============================================================================

"""
    build_dof_connectivity(elements, n_total_dofs) -> DOFConnectivity
    build_dof_connectivity(elements, dof_handler::DOFHandler) -> DOFConnectivity

Build the DOF→element inverse mapping in a single pass over `elements`.

Complexity is `O(n_elements × dofs_per_element)`. After construction the
returned `DOFConnectivity` supports zero-allocation iteration through
`dof_to_elements[dof_i]`. The `DOFHandler` overload is a thin convenience
that forwards `dof_handler.total_dofs`.

# Example
```julia
S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
ElemType = Element{Tetrahedron{4}, Lagrange{1}, S}
elements, handler = create_elements!(mesh, ElemType)

connectivity = build_dof_connectivity(elements, handler)
for dof_i in 1:connectivity.n_total_dofs
    for conn in connectivity.dof_to_elements[dof_i]
        elem = elements[conn.elem_id]
        local_dof = conn.local_dof_idx
        # ...
    end
end
```
"""
function build_dof_connectivity(elements::Vector, n_total_dofs::Integer)
    n_total_dofs = Int(n_total_dofs)

    if n_total_dofs == 0
        return DOFConnectivity(Vector{DOFElementConnection}[], 0)
    end

    # One bucket per DOF. Pre-size each to the typical 3D fan-in (~6 elements
    # per DOF) so the inner push! loop avoids most reallocations.
    dof_to_elements = [DOFElementConnection[] for _ in 1:n_total_dofs]
    avg_connections = 6
    for vec in dof_to_elements
        sizehint!(vec, avg_connections)
    end

    for (elem_id, elem) in enumerate(elements)
        dofs = element_dofs(elem)  # NTuple{N, UInt64}
        for (local_idx, global_dof) in enumerate(dofs)
            if global_dof < 1 || global_dof > n_total_dofs
                error("Invalid global DOF index $global_dof (must be in [1, $n_total_dofs])")
            end
            push!(dof_to_elements[global_dof], DOFElementConnection(elem_id, local_idx))
        end
    end

    return DOFConnectivity(dof_to_elements, n_total_dofs)
end

# Convenience overload: pull `n_total_dofs` straight from a DOFHandler so
# callers do not have to thread it through.
build_dof_connectivity(elements::Vector, dof_handler::DOFHandler) =
    build_dof_connectivity(elements, dof_handler.total_dofs)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

"""
    connection_count(connectivity::DOFConnectivity, dof_i::Int) -> Int

Get number of elements touching DOF i.

Zero-allocation: Direct array length access.
"""
@inline connection_count(connectivity::DOFConnectivity, dof_i::Int) =
    length(connectivity.dof_to_elements[dof_i])

"""
    is_empty(connectivity::DOFConnectivity, dof_i::Int) -> Bool

Check if DOF i has no connections.

Zero-allocation: Direct array length check.
"""
@inline is_empty(connectivity::DOFConnectivity, dof_i::Int) =
    isempty(connectivity.dof_to_elements[dof_i])

# Exports for these types live in `src/exports.jl`.
