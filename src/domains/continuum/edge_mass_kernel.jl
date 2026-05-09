# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

#=
Diagonal **edge mass** on global `Edge` DOFs for [`Mesh{8, Hex8}`](@ref),
[`Mesh{20, Hex20}`](@ref), [`Mesh{4, Tet4}`](@ref), [`Mesh{10, Tet10}`](@ref),
[`Mesh{6, Wedge6}`](@ref), or [`Mesh{5, Pyr5}`](@ref).

Each element contributes `length(edge) / n_patch` for coincident local edges,
with `n_patch` the number of volume elements sharing that mesh edge.

This mirrors [`FacetMassKernel`](@ref) and exercises [`Edge`](@ref) entities in
`local_dof_layout` together with [`AbstractFacetConnectivityMaps`](@ref).
=#

using ..JuliaFEM: AbstractKernel, AssemblyMaterialWorkspace, Mesh,
                  Hex8, Hex20, Tet4, Tet10, Wedge6, Pyr5
using ..JuliaFEM: AbstractFacetConnectivityMaps, Hex8FacetMaps, Tet4FacetMaps,
                  Wedge6FacetMaps, Pyr5FacetMaps
using ..JuliaFEM: hex8_edge_length_physical, tet_edge_length_physical, wedge_edge_length_physical,
                  pyr_edge_length_physical
using ..JuliaFEM: build_hex8_facet_maps, build_hex20_facet_maps, build_tet4_facet_maps,
                  build_tet10_facet_maps, build_wedge6_facet_maps, build_pyr5_facet_maps
import ..JuliaFEM: qpoint_buffer_eltype, update_qpoint_buffer!, evaluate_entry,
                   evaluate_mass_entry,
                   reference_fields, get_field, dofs_per_node
using ..JuliaFEM: DOFLayoutEntry, field_idx, entity_local, component


"""
    EdgeMassKernel

Edge unknowns (`DOF{Float64, Edge}` or `DOF{Vec{k}, Edge}` with independent components).
Assembles `K_ee[e,e] += length(e) / patch_multiplicity` per element visit.

Construct from a mesh so edge numbering matches [`DOFHandler`](@ref):

```julia
mesh = create_unit_cube_mesh(Hex8)
S = @DOFSet{circ::DOF{Float64, Edge}}
kernel = EdgeMassKernel(mesh)
elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
```
"""
struct EdgeMassKernel <: AbstractKernel
    facet_maps::AbstractFacetConnectivityMaps
end

function EdgeMassKernel(mesh::Mesh{8, Hex8})
    return EdgeMassKernel(build_hex8_facet_maps(mesh))
end

function EdgeMassKernel(mesh::Mesh{20, Hex20})
    return EdgeMassKernel(build_hex20_facet_maps(mesh))
end

function EdgeMassKernel(mesh::Mesh{4, Tet4})
    return EdgeMassKernel(build_tet4_facet_maps(mesh))
end

function EdgeMassKernel(mesh::Mesh{10, Tet10})
    return EdgeMassKernel(build_tet10_facet_maps(mesh))
end

function EdgeMassKernel(mesh::Mesh{6, Wedge6})
    return EdgeMassKernel(build_wedge6_facet_maps(mesh))
end

function EdgeMassKernel(mesh::Mesh{5, Pyr5})
    return EdgeMassKernel(build_pyr5_facet_maps(mesh))
end

function get_field(::EdgeMassKernel)
    error("EdgeMassKernel uses Edge DOFs — use `local_dof_layout(E)` and `elem.dof_indices`.")
end

@inline dofs_per_node(::EdgeMassKernel) = 1

@inline qpoint_buffer_eltype(::EdgeMassKernel) = Float64

@inline function reference_fields(::EdgeMassKernel)
    return ((aux = 0.0,), NamedTuple())
end

@inline function update_qpoint_buffer!(
    ::AbstractVector{Float64},
    ::AssemblyMaterialWorkspace,
    ::EdgeMassKernel,
)
    return nothing
end

@inline function evaluate_entry(
    kernel::EdgeMassKernel,
    geometry_cache,
    ::AbstractVector{Float64},
    layout_i::DOFLayoutEntry,
    layout_j::DOFLayoutEntry,
    elem_id::Int,
)
    fi = field_idx(layout_i)
    fj = field_idx(layout_j)
    fi == fj || return 0.0
    ei = entity_local(layout_i)
    ej = entity_local(layout_j)
    ei == ej || return 0.0
    component(layout_i) == component(layout_j) || return 0.0

    maps = kernel.facet_maps
    X = geometry_cache.X
    if maps isa Hex8FacetMaps
        len = hex8_edge_length_physical(X, ei)
    elseif maps isa Tet4FacetMaps
        len = tet_edge_length_physical(X, ei)
    elseif maps isa Wedge6FacetMaps
        len = wedge_edge_length_physical(X, ei)
    elseif maps isa Pyr5FacetMaps
        len = pyr_edge_length_physical(X, ei)
    else
        error("EdgeMassKernel: unsupported facet maps type $(typeof(maps)).")
    end
    frac = maps.elem_edge_fraction[ei, elem_id]
    return len * frac
end

@inline evaluate_mass_entry(
    ::EdgeMassKernel,
    geometry_cache,
    qp_buffer,
    layout_i::DOFLayoutEntry,
    layout_j::DOFLayoutEntry,
) = 0.0
