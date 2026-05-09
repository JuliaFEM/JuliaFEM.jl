# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using Tensors

"""
    Tet4FacetMaps

Like [`Hex8FacetMaps`](@ref) for tetrahedra: global edge and triangular face ids on
`Mesh{4, Tet4}` or **`Mesh{10, Tet10}`** (corner-only skeleton, same `edges`/`faces`
local numbering). Patch fractions and orientation hints target lowest-order facet DOFs.

# Fields
Same pattern as [`Hex8FacetMaps`](@ref): `elem_edge_gid` `(6, nelem)`,
`elem_face_gid` `(4, nelem)`, `elem_edge_orientation`, `elem_face_orientation`,
fraction matrices.
"""
struct Tet4FacetMaps <: AbstractFacetConnectivityMaps
    n_edges::Int
    n_faces::Int
    elem_edge_gid::Matrix{Int}
    elem_face_gid::Matrix{Int}
    elem_edge_orientation::Matrix{Int8}
    elem_face_orientation::Matrix{Int8}
    elem_edge_fraction::Matrix{Float64}
    elem_face_fraction::Matrix{Float64}
end

@inline function _sorted_face_triplet(conn::NTuple{N, UInt32}, verts::NTuple{3, Int}) where {N}
    v = (conn[verts[1]], conn[verts[2]], conn[verts[3]])
    x = Int[v[1], v[2], v[3]]
    sort!(x)
    return (UInt32(x[1]), UInt32(x[2]), UInt32(x[3]))
end

"""
    tet_face_area_physical(X, local_face::Int) -> Float64

Triangle area (`\\frac{1}{2}\\|(p_2-p_1)\\times(p_3-p_1)\\|`) for tet face
`local_face` using corner indices only (`length(X) ∈ {4, 10}`).
"""
function tet_face_area_physical(X::AbstractVector{V}, local_face::Int) where {V<:Vec{3}}
    fc = faces(Tet4())[local_face]
    @inbounds p1 = X[fc.vertices[1]]
    @inbounds p2 = X[fc.vertices[2]]
    @inbounds p3 = X[fc.vertices[3]]
    return 0.5 * norm((p2 - p1) × (p3 - p1))
end

"""
    tet_edge_length_physical(X, local_edge::Int) -> Float64

Edge length for tet skeleton edge (`edges(::Tet4)`), corner endpoints (`length(X) ∈ {4, 10}`).
"""
function tet_edge_length_physical(X::AbstractVector{V}, local_edge::Int) where {V<:Vec{3}}
    ed = edges(Tet4())[local_edge]
    i, j = ed.vertices
    @inbounds return norm(X[i] - X[j])
end

"""
    tet_face_outward_sign(X, local_face::Int) -> Int8

Triangle face `local_face ∈ 1:4` on a tet: compare `(p2-p1)×(p3-p1)` with the direction
from face centroid toward the element centroid (`mean` of all nodes in `X`, so `Tet10`
includes mid-edge nodes in the volume reference).
"""
function tet_face_outward_sign(X::AbstractVector{V}, local_face::Int) where {V<:Vec{3}}
    fc = faces(Tet4())[local_face]
    vs = fc.vertices
    @inbounds p1 = X[vs[1]]
    @inbounds p2 = X[vs[2]]
    @inbounds p3 = X[vs[3]]
    n = (p2 - p1) × (p3 - p1)
    cf = (p1 + p2 + p3) / 3
    ce = _elem_centroid_mean(X)
    to_centroid = ce - cf
    s = dot(n, to_centroid)
    return s < 0 ? Int8(1) : Int8(-1)
end

@inline function _tet4_element_coords(mesh::Mesh{4, Tet4}, conn::NTuple{4, UInt32})
    return Vec{3, Float64}[mesh.nodes[Int(conn[i])] for i in 1:4]
end

"""
    build_tet4_facet_maps(mesh::Mesh{4, Tet4}) -> Tet4FacetMaps
"""
function build_tet4_facet_maps(mesh::Mesh{4, Tet4})
    Ktop = Tet4()
    edge_defs = edges(Ktop)
    face_defs = faces(Ktop)
    nelem = length(mesh.connectivity)

    edge_dict = Dict{Tuple{UInt32, UInt32}, Int}()
    face_dict = Dict{NTuple{3, UInt32}, Int}()

    elem_edge_gid = Matrix{Int}(undef, 6, nelem)
    elem_face_gid = Matrix{Int}(undef, 4, nelem)
    elem_edge_orientation = Matrix{Int8}(undef, 6, nelem)

    next_edge = 1
    next_face = 1

    @inbounds for eid in 1:nelem
        conn = mesh.connectivity[eid]

        for le in 1:6
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

        for lf in 1:4
            fc = face_defs[lf]
            fk = _sorted_face_triplet(conn, fc.vertices)
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
        for le in 1:6
            edge_touch[elem_edge_gid[le, eid]] += 1
        end
        for lf in 1:4
            face_touch[elem_face_gid[lf, eid]] += 1
        end
    end

    elem_edge_fraction = Matrix{Float64}(undef, 6, nelem)
    elem_face_fraction = Matrix{Float64}(undef, 4, nelem)
    elem_face_orientation = Matrix{Int8}(undef, 4, nelem)

    @inbounds for eid in 1:nelem
        conn = mesh.connectivity[eid]
        X = _tet4_element_coords(mesh, conn)
        for le in 1:6
            g = elem_edge_gid[le, eid]
            elem_edge_fraction[le, eid] = 1.0 / edge_touch[g]
        end
        for lf in 1:4
            g = elem_face_gid[lf, eid]
            elem_face_fraction[lf, eid] = 1.0 / face_touch[g]
            elem_face_orientation[lf, eid] = tet_face_outward_sign(X, lf)
        end
    end

    return Tet4FacetMaps(
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
    tet_facet_gid_from_corners(mesh::Mesh{4,Tet4}, maps::Tet4FacetMaps, corners::NTuple{3,Int}) -> Int

Global triangular facet id (`1 … maps.n_faces`) for mesh nodes `corners`, using the same
sorted-node key as [`build_tet4_facet_maps`](@ref). Node order in `corners` may be permuted.

Returns `0` if no element carries that triangle (wrong mesh / typo).

Intended for preprocessing and tests (linear scan over elements). Hot assembly paths should
use [`Tet4FacetMaps`](@ref).`elem_face_gid` directly.
"""
function tet_facet_gid_from_corners(
    mesh::Mesh{4, Tet4},
    maps::Tet4FacetMaps,
    corners::NTuple{3, Int},
)
    Ktop = Tet4()
    face_defs = faces(Ktop)
    x = Int[Int(corners[1]), Int(corners[2]), Int(corners[3])]
    sort!(x)
    fk_target = (UInt32(x[1]), UInt32(x[2]), UInt32(x[3]))

    @inbounds for eid in 1:length(mesh.connectivity)
        conn = mesh.connectivity[eid]
        for lf in 1:4
            fc = face_defs[lf]
            fk = _sorted_face_triplet(conn, fc.vertices)
            if fk == fk_target
                return maps.elem_face_gid[lf, eid]
            end
        end
    end
    return 0
end

@inline function _tet10_element_coords(mesh::Mesh{10, Tet10}, conn::NTuple{10, UInt32})
    return Vec{3, Float64}[mesh.nodes[Int(conn[i])] for i in 1:10]
end

"""
    build_tet10_facet_maps(mesh::Mesh{10, Tet10}) -> Tet4FacetMaps

Same skeleton as [`build_tet4_facet_maps`](@ref); corners use local indices `1:4`
into each connectivity row.
"""
function build_tet10_facet_maps(mesh::Mesh{10, Tet10})
    Ktop = Tet4()
    edge_defs = edges(Ktop)
    face_defs = faces(Ktop)
    nelem = length(mesh.connectivity)

    edge_dict = Dict{Tuple{UInt32, UInt32}, Int}()
    face_dict = Dict{NTuple{3, UInt32}, Int}()

    elem_edge_gid = Matrix{Int}(undef, 6, nelem)
    elem_face_gid = Matrix{Int}(undef, 4, nelem)
    elem_edge_orientation = Matrix{Int8}(undef, 6, nelem)

    next_edge = 1
    next_face = 1

    @inbounds for eid in 1:nelem
        conn = mesh.connectivity[eid]

        for le in 1:6
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

        for lf in 1:4
            fc = face_defs[lf]
            fk = _sorted_face_triplet(conn, fc.vertices)
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
        for le in 1:6
            edge_touch[elem_edge_gid[le, eid]] += 1
        end
        for lf in 1:4
            face_touch[elem_face_gid[lf, eid]] += 1
        end
    end

    elem_edge_fraction = Matrix{Float64}(undef, 6, nelem)
    elem_face_fraction = Matrix{Float64}(undef, 4, nelem)
    elem_face_orientation = Matrix{Int8}(undef, 4, nelem)

    @inbounds for eid in 1:nelem
        conn = mesh.connectivity[eid]
        X = _tet10_element_coords(mesh, conn)
        for le in 1:6
            g = elem_edge_gid[le, eid]
            elem_edge_fraction[le, eid] = 1.0 / edge_touch[g]
        end
        for lf in 1:4
            g = elem_face_gid[lf, eid]
            elem_face_fraction[lf, eid] = 1.0 / face_touch[g]
            elem_face_orientation[lf, eid] = tet_face_outward_sign(X, lf)
        end
    end

    return Tet4FacetMaps(
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
    tet4_edge_orientation_sign(conn::NTuple{4,UInt32}, local_edge::Int) -> Int8

Directed-edge hint matching [`build_tet4_facet_maps`](@ref) / `elem_edge_orientation`.
"""
function tet4_edge_orientation_sign(conn::NTuple{4, UInt32}, local_edge::Int)
    ed = edges(Tet4())[local_edge]
    va, vb = ed.vertices
    na = Int(conn[va])
    nb = Int(conn[vb])
    return na < nb ? Int8(1) : Int8(-1)
end

"""
    tet10_edge_orientation_sign(conn::NTuple{10,UInt32}, local_edge::Int) -> Int8

Same as [`tet4_edge_orientation_sign`](@ref), using corner indices `1:4` into `Tet10` connectivity.
"""
function tet10_edge_orientation_sign(conn::NTuple{10, UInt32}, local_edge::Int)
    ed = edges(Tet10())[local_edge]
    va, vb = ed.vertices
    na = Int(conn[va])
    nb = Int(conn[vb])
    return na < nb ? Int8(1) : Int8(-1)
end
