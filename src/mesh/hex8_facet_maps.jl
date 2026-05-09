# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using Tensors

"""
    AbstractFacetConnectivityMaps

Supertype for per-volume-element facet connectivity (`elem_edge_gid`,
`elem_face_gid`, fractions, orientation hints). Used by [`DOFHandler`](@ref)
for [`Edge`](@ref) / [`Face`](@ref) fields on conforming meshes.

Concrete types: [`Hex8FacetMaps`](@ref), [`Tet4FacetMaps`](@ref), [`Wedge6FacetMaps`](@ref),
[`Pyr5FacetMaps`](@ref).

Future hp / variable facet multiplicity: conforming meshes must agree on the
number of DOFs on each shared edge or face. When counts differ per global edge,
global numbering can use a CSR-style offset vector; see [`edge_dof_csr_offsets`](@ref).
"""
abstract type AbstractFacetConnectivityMaps end

"""
    edge_dof_csr_offsets(maps::AbstractFacetConnectivityMaps)

If facet scalar DOF counts vary per global mesh edge, implementations may return a
length-`(n_edges + 1)` vector of one-based offsets into a contiguous edge-DOF pool
(so global DOFs on edge `e` occupy a variable-width band). Return `nothing` when
each edge owns a fixed-width block derived only from `dof_size` of the field (current default).

Not wired into [`DOFHandler`](@ref) yet; this is the intended extension point for hp-style numbering.
"""
edge_dof_csr_offsets(::AbstractFacetConnectivityMaps) = nothing

"""
    Hex8FacetMaps

Mesh-wide numbering of topological **edges** and **faces** for a conforming
`Mesh{8, Hex8}` or **`Mesh{20, Hex20}`** (same corner-only skeleton as `Hex8`):
each undirected edge / sorted quad of **corner** vertices receives one global id.
Element-local indices follow `edges(::Hex8)` and `faces(::Hex8)` (corners `1:8`
into element connectivity).

Used by [`DOFHandler`](@ref) when a [`DOFSet`](@ref) places unknowns on
[`Edge`](@ref) or [`Face`](@ref) (e.g. lowest-order Raviart–Thomas face fluxes,
Nédélec edge circulations).

# Fields
- `n_edges`, `n_faces`: global counts
- `elem_edge_gid::Matrix{Int}` — shape `(12, nelem)`, global edge id `≥ 1`
- `elem_face_gid::Matrix{Int}` — shape `(6, nelem)`, global face id `≥ 1`
- `elem_edge_fraction`, `elem_face_fraction`: same shape, `1 / patch_multiplicity`
  so volumetric assembly patterns that sum element contributions recover a
  **partition of unity** on shared facets (`2` interior, `1` boundary for a
  conforming hex brick mesh).
- `elem_edge_orientation::Matrix{Int8}` — shape `(12, nelem)`, `±1` indicating
  whether the directed local edge (`edges(::Hex8)` vertex ordering) runs from
  the lower global node id toward the higher (`+1`) or the reverse (`-1`).
  On shared edges, adjacent elements often (not always) store opposite values;
  pairs that agree need an extra mesh-dependent flip in a full Nédélec kernel.
- `elem_face_orientation::Matrix{Int8}` — shape `(6, nelem)`, `±1` comparing the
  cyclic cross product from [`faces(::Hex8)`](@ref) vertex order with the
  **outward** direction (via [`hex8_face_outward_sign`](@ref))). Useful for
  `RT₀`-style normal flux DOFs before full Piola assembly exists.
"""
struct Hex8FacetMaps <: AbstractFacetConnectivityMaps
    n_edges::Int
    n_faces::Int
    elem_edge_gid::Matrix{Int}
    elem_face_gid::Matrix{Int}
    elem_edge_orientation::Matrix{Int8}
    elem_face_orientation::Matrix{Int8}
    elem_edge_fraction::Matrix{Float64}
    elem_face_fraction::Matrix{Float64}
end

@inline function _sorted_edge_pair(a::UInt32, b::UInt32)
    return a <= b ? (a, b) : (b, a)
end

"""Arithmetic mean of element node coordinates (`X[i]` for all `i`)."""
@inline function _elem_centroid_mean(X::AbstractVector{V}) where {V<:Vec{3}}
    @inbounds v = X[1]
    @inbounds for i in 2:length(X)
        v += X[i]
    end
    return v / length(X)
end

function _sorted_face_quad(conn::NTuple{N, UInt32}, verts::NTuple{4, Int}) where {N}
    v = (
        conn[verts[1]],
        conn[verts[2]],
        conn[verts[3]],
        conn[verts[4]],
    )
    x = Int[v[1], v[2], v[3], v[4]]
    sort!(x)
    return (UInt32(x[1]), UInt32(x[2]), UInt32(x[3]), UInt32(x[4]))
end

"""
    build_hex8_facet_maps(mesh::Mesh{8, Hex8}) -> Hex8FacetMaps

Build unique edge and face ids for a conforming Hex8 mesh (structured or not).
"""
function build_hex8_facet_maps(mesh::Mesh{8, Hex8})
    Ktop = Hex8()
    edge_defs = edges(Ktop)
    face_defs = faces(Ktop)
    nelem = length(mesh.connectivity)

    edge_dict = Dict{Tuple{UInt32, UInt32}, Int}()
    face_dict = Dict{NTuple{4, UInt32}, Int}()

    elem_edge_gid = Matrix{Int}(undef, 12, nelem)
    elem_face_gid = Matrix{Int}(undef, 6, nelem)
    elem_edge_orientation = Matrix{Int8}(undef, 12, nelem)

    next_edge = 1
    next_face = 1

    @inbounds for eid in 1:nelem
        conn = mesh.connectivity[eid]

        for le in 1:12
            ed = edge_defs[le]
            i, j = ed.vertices
            ek = _sorted_edge_pair(conn[i], conn[j])
            gid = get(edge_dict, ek, nothing)
            if gid === nothing
                gid = next_edge
                edge_dict[ek] = gid
                next_edge += 1
            end
            elem_edge_gid[le, eid] = gid
            na = Int(conn[i])
            nb = Int(conn[j])
            elem_edge_orientation[le, eid] = na < nb ? Int8(1) : Int8(-1)
        end

        for lf in 1:6
            fc = face_defs[lf]
            fk = _sorted_face_quad(conn, fc.vertices)
            gid = get(face_dict, fk, nothing)
            if gid === nothing
                gid = next_face
                face_dict[fk] = gid
                next_face += 1
            end
            elem_face_gid[lf, eid] = gid
        end
    end

    n_edges = next_edge - 1
    n_faces = next_face - 1

    edge_touch = zeros(Int, n_edges)
    face_touch = zeros(Int, n_faces)
    @inbounds for eid in 1:nelem
        for le in 1:12
            edge_touch[elem_edge_gid[le, eid]] += 1
        end
        for lf in 1:6
            face_touch[elem_face_gid[lf, eid]] += 1
        end
    end

    elem_edge_fraction = Matrix{Float64}(undef, 12, nelem)
    elem_face_fraction = Matrix{Float64}(undef, 6, nelem)
    elem_face_orientation = Matrix{Int8}(undef, 6, nelem)
    @inbounds for eid in 1:nelem
        conn = mesh.connectivity[eid]
        X = _hex8_element_coords(mesh, conn)
        for le in 1:12
            g = elem_edge_gid[le, eid]
            elem_edge_fraction[le, eid] = 1.0 / edge_touch[g]
        end
        for lf in 1:6
            g = elem_face_gid[lf, eid]
            elem_face_fraction[lf, eid] = 1.0 / face_touch[g]
            elem_face_orientation[lf, eid] = hex8_face_outward_sign(X, lf)
        end
    end

    return Hex8FacetMaps(
        n_edges,
        n_faces,
        elem_edge_gid,
        elem_face_gid,
        elem_edge_orientation,
        elem_face_orientation,
        elem_edge_fraction,
        elem_face_fraction,
    )
end

"""
    build_hex20_facet_maps(mesh::Mesh{20, Hex20}) -> Hex8FacetMaps

Same edge/face skeleton and storage as [`build_hex8_facet_maps`](@ref); corner
vertices use local indices `1:8` into each `20`-tuple connectivity row.
"""
function build_hex20_facet_maps(mesh::Mesh{20, Hex20})
    Ktop = Hex8()
    edge_defs = edges(Ktop)
    face_defs = faces(Ktop)
    nelem = length(mesh.connectivity)

    edge_dict = Dict{Tuple{UInt32, UInt32}, Int}()
    face_dict = Dict{NTuple{4, UInt32}, Int}()

    elem_edge_gid = Matrix{Int}(undef, 12, nelem)
    elem_face_gid = Matrix{Int}(undef, 6, nelem)
    elem_edge_orientation = Matrix{Int8}(undef, 12, nelem)

    next_edge = 1
    next_face = 1

    @inbounds for eid in 1:nelem
        conn = mesh.connectivity[eid]

        for le in 1:12
            ed = edge_defs[le]
            i, j = ed.vertices
            ek = _sorted_edge_pair(conn[i], conn[j])
            gid = get(edge_dict, ek, nothing)
            if gid === nothing
                gid = next_edge
                edge_dict[ek] = gid
                next_edge += 1
            end
            elem_edge_gid[le, eid] = gid
            na = Int(conn[i])
            nb = Int(conn[j])
            elem_edge_orientation[le, eid] = na < nb ? Int8(1) : Int8(-1)
        end

        for lf in 1:6
            fc = face_defs[lf]
            fk = _sorted_face_quad(conn, fc.vertices)
            gid = get(face_dict, fk, nothing)
            if gid === nothing
                gid = next_face
                face_dict[fk] = gid
                next_face += 1
            end
            elem_face_gid[lf, eid] = gid
        end
    end

    n_edges = next_edge - 1
    n_faces = next_face - 1

    edge_touch = zeros(Int, n_edges)
    face_touch = zeros(Int, n_faces)
    @inbounds for eid in 1:nelem
        for le in 1:12
            edge_touch[elem_edge_gid[le, eid]] += 1
        end
        for lf in 1:6
            face_touch[elem_face_gid[lf, eid]] += 1
        end
    end

    elem_edge_fraction = Matrix{Float64}(undef, 12, nelem)
    elem_face_fraction = Matrix{Float64}(undef, 6, nelem)
    elem_face_orientation = Matrix{Int8}(undef, 6, nelem)

    @inbounds for eid in 1:nelem
        conn = mesh.connectivity[eid]
        X = _hex20_element_coords(mesh, conn)
        for le in 1:12
            g = elem_edge_gid[le, eid]
            elem_edge_fraction[le, eid] = 1.0 / edge_touch[g]
        end
        for lf in 1:6
            g = elem_face_gid[lf, eid]
            elem_face_fraction[lf, eid] = 1.0 / face_touch[g]
            elem_face_orientation[lf, eid] = hex8_face_outward_sign(X, lf)
        end
    end

    return Hex8FacetMaps(
        n_edges,
        n_faces,
        elem_edge_gid,
        elem_face_gid,
        elem_edge_orientation,
        elem_face_orientation,
        elem_edge_fraction,
        elem_face_fraction,
    )
end

@inline function _hex8_element_coords(mesh::Mesh{8, Hex8}, conn::NTuple{8, UInt32})
    return Vec{3, Float64}[
        mesh.nodes[Int(conn[i])] for i in 1:8
    ]
end

@inline function _hex20_element_coords(mesh::Mesh{20, Hex20}, conn::NTuple{20, UInt32})
    return Vec{3, Float64}[mesh.nodes[Int(conn[i])] for i in 1:20]
end

"""
    hex8_edge_orientation_sign(conn::NTuple{8,UInt32}, local_edge::Int) -> Int8

Return `±1`: directed local edge `local_edge` runs from smaller global node
id toward larger (`+1`) or the reverse (`-1`). Matches
`elem_edge_orientation[local_edge, elem_id]` from [`build_hex8_facet_maps`](@ref).
"""
function hex8_edge_orientation_sign(conn::NTuple{8, UInt32}, local_edge::Int)
    ed = edges(Hex8())[local_edge]
    va, vb = ed.vertices
    na = Int(conn[va])
    nb = Int(conn[vb])
    return na < nb ? Int8(1) : Int8(-1)
end

"""
    hex20_edge_orientation_sign(conn::NTuple{20,UInt32}, local_edge::Int) -> Int8

Same convention as [`hex8_edge_orientation_sign`](@ref), using corner indices `1:8`
into `Hex20` connectivity.
"""
function hex20_edge_orientation_sign(conn::NTuple{20, UInt32}, local_edge::Int)
    ed = edges(Hex20())[local_edge]
    va, vb = ed.vertices
    na = Int(conn[va])
    nb = Int(conn[vb])
    return na < nb ? Int8(1) : Int8(-1)
end

"""
    hex8_face_outward_sign(X, local_face::Int) -> Int8

Given element node coordinates `X` (`Vec{3}` per node — corners `1:8` are used for
the face geometry; additional nodes e.g. on `Hex20` affect only the volume centroid),
compare the normal from the
first two face edges (vertex order in [`faces(::Hex8)`](@ref)) with the vector
from the face centroid to the element centroid. Returns `+1` when they agree up
to positive scaling (outward), `-1` when the cyclic ordering defines an inward
normal relative to the hex volume.

Degenerate configurations (`dot ≈ 0`) return `-1`.
"""
function hex8_face_outward_sign(X::AbstractVector{V}, local_face::Int) where {V<:Vec{3}}
    fc = faces(Hex8())[local_face]
    vs = fc.vertices
    @inbounds p1 = X[vs[1]]
    @inbounds p2 = X[vs[2]]
    @inbounds p3 = X[vs[3]]
    @inbounds p4 = X[vs[4]]
    n = (p2 - p1) × (p3 - p1)
    cf = 0.25 * (p1 + p2 + p3 + p4)
    @inbounds ce = _elem_centroid_mean(X)
    to_centroid = ce - cf
    s = dot(n, to_centroid)
    return s < 0 ? Int8(1) : Int8(-1)
end

"""
    hex8_face_area_physical(X, local_face::Int) -> Float64

Physical area of Hex8 face `local_face ∈ 1:6` from corner coordinates `X[face verts]`
(`Vec{3}`; typically `length(X) ∈ {8, 20}`), splitting the quad into two triangles.
"""
function hex8_face_area_physical(X::AbstractVector{V}, local_face::Int) where {V<:Vec{3}}
    fc = faces(Hex8())[local_face]
    vs = fc.vertices
    @inbounds p1 = X[vs[1]]
    @inbounds p2 = X[vs[2]]
    @inbounds p3 = X[vs[3]]
    @inbounds p4 = X[vs[4]]
    c1 = (p2 - p1) × (p3 - p1)
    c2 = (p3 - p1) × (p4 - p1)
    return 0.5 * (norm(c1) + norm(c2))
end

"""
    hex8_edge_length_physical(X, local_edge::Int) -> Float64

Euclidean length of Hex8 skeleton edge `local_edge ∈ 1:12` between corner
indices (`length(X) ∈ {8, 20}` for `Hex8` / `Hex20`).
"""
function hex8_edge_length_physical(X::AbstractVector{V}, local_edge::Int) where {V<:Vec{3}}
    ed = edges(Hex8())[local_edge]
    i, j = ed.vertices
    @inbounds return norm(X[i] - X[j])
end
