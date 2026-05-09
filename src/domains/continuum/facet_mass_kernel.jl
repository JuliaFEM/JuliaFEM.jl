# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

#=
Diagonal **facet mass** on global `Face` DOFs for [`Mesh{8, Hex8}`](@ref),
[`Mesh{20, Hex20}`](@ref), [`Mesh{4, Tet4}`](@ref), [`Mesh{10, Tet10}`](@ref),
[`Mesh{6, Wedge6}`](@ref), or [`Mesh{5, Pyr5}`](@ref).

Each element contributes `area(face) / n_patch` to the diagonal entry of
every local face unknown, where `n_patch` is the number of volume elements
sharing that facet (`2` interior, `1` boundary on a conforming hex mesh).
Summing two element contributions on an interior face reproduces the full
facet area — a smoke test for facet maps + [`DOFHandler`](@ref) face numbering.

This is **not** a volume-integrated mass matrix; it exists to exercise
`Face` entities in `local_dof_layout`, assembly, and the extended
`evaluate_entry(..., elem_id)` surface.

See [`FacetMassKernel`](@ref).
=#

using ..JuliaFEM: AbstractKernel, AssemblyMaterialWorkspace, Mesh,
                  Hex8, Hex20, Tet4, Tet10, Wedge6, Pyr5
using ..JuliaFEM: AbstractFacetConnectivityMaps, Hex8FacetMaps, Tet4FacetMaps,
                  Wedge6FacetMaps, Pyr5FacetMaps
using ..JuliaFEM: hex8_face_area_physical, tet_face_area_physical, wedge_face_area_physical,
                  pyr_face_area_physical
using ..JuliaFEM: build_hex8_facet_maps, build_hex20_facet_maps, build_tet4_facet_maps,
                  build_tet10_facet_maps, build_wedge6_facet_maps, build_pyr5_facet_maps
import ..JuliaFEM: qpoint_buffer_eltype, update_qpoint_buffer!, evaluate_entry,
                   evaluate_mass_entry,
                   reference_fields, get_field, dofs_per_node
using ..JuliaFEM: DOFLayoutEntry, field_idx, entity_local, component


"""
    FacetMassKernel

Face unknowns (`DOF{Float64, Face}` or `DOF{Vec{k}, Face}` with independent components).
Assembles `K_ff[f,f] += area(f) / patch_multiplicity` per element visit.

Construct from a mesh so facet numbering matches [`DOFHandler`](@ref):

```julia
mesh = create_unit_cube_mesh(Hex8)
S = @DOFSet{flux::DOF{Float64, Face}}
kernel = FacetMassKernel(mesh)
elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
```
"""
struct FacetMassKernel <: AbstractKernel
    facet_maps::AbstractFacetConnectivityMaps
end

function FacetMassKernel(mesh::Mesh{8, Hex8})
    return FacetMassKernel(build_hex8_facet_maps(mesh))
end

function FacetMassKernel(mesh::Mesh{20, Hex20})
    return FacetMassKernel(build_hex20_facet_maps(mesh))
end

function FacetMassKernel(mesh::Mesh{4, Tet4})
    return FacetMassKernel(build_tet4_facet_maps(mesh))
end

function FacetMassKernel(mesh::Mesh{10, Tet10})
    return FacetMassKernel(build_tet10_facet_maps(mesh))
end

function FacetMassKernel(mesh::Mesh{6, Wedge6})
    return FacetMassKernel(build_wedge6_facet_maps(mesh))
end

function FacetMassKernel(mesh::Mesh{5, Pyr5})
    return FacetMassKernel(build_pyr5_facet_maps(mesh))
end

function get_field(::FacetMassKernel)
    error("FacetMassKernel uses Face DOFs — use `local_dof_layout(E)` and `elem.dof_indices`.")
end

@inline dofs_per_node(::FacetMassKernel) = 1

@inline qpoint_buffer_eltype(::FacetMassKernel) = Float64

@inline function reference_fields(::FacetMassKernel)
    return ((aux = 0.0,), NamedTuple())
end

@inline function update_qpoint_buffer!(
    ::AbstractVector{Float64},
    ::AssemblyMaterialWorkspace,
    ::FacetMassKernel,
)
    return nothing
end

@inline function evaluate_entry(
    kernel::FacetMassKernel,
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
        area = hex8_face_area_physical(X, ei)
    elseif maps isa Tet4FacetMaps
        area = tet_face_area_physical(X, ei)
    elseif maps isa Wedge6FacetMaps
        area = wedge_face_area_physical(X, ei)
    elseif maps isa Pyr5FacetMaps
        area = pyr_face_area_physical(X, ei)
    else
        error("FacetMassKernel: unsupported facet maps type $(typeof(maps)).")
    end
    frac = maps.elem_face_fraction[ei, elem_id]
    return area * frac
end

@inline evaluate_mass_entry(
    ::FacetMassKernel,
    geometry_cache,
    qp_buffer,
    layout_i::DOFLayoutEntry,
    layout_j::DOFLayoutEntry,
) = 0.0
