# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Shared helpers for small `Hex8` brick regression tests (structured mesh,
boundary DOF collection, interior load DOF).
"""

using JuliaFEM

function brick_hex_mesh(nx::Int, ny::Int, nz::Int)
    return create_structured_box_mesh(Hex8;
        xmin = 0.0, xmax = 1.0, nx = nx,
        ymin = 0.0, ymax = 1.0, ny = ny,
        zmin = 0.0, zmax = 1.0, nz = nz,
    )
end

function collect_vertex_dofs_on_nodeset(handler, mesh, set_sym::Symbol)
    d = Int[]
    for nid in get_nodes_in_set(mesh, set_sym)
        append!(d, get_node_dofs(handler, Int(nid)))
    end
    sort!(unique!(d))
    return d
end

function interior_uz_dof_index(handler, mesh; x_threshold::Float64 = 0.25)
    for i in 1:length(mesh.nodes)
        mesh.nodes[i][1] > x_threshold || continue
        nd = get_node_dofs(handler, i)
        return Int(nd[3])
    end
    error("no interior node for load (x_threshold = $x_threshold)")
end

"""First global DOF index for `field_idx` on `elem_id` (cell-centred fields)."""
@inline function first_field_dof(handler, field_idx::Int, elem_id::Int = 1)
    return handler.field_starts[field_idx][elem_id]
end
