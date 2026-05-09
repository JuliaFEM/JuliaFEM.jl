# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using Tensors

"""
    Pyr5FacetMaps

Global edge and face ids on `Mesh{5, Pyr5}`: one quadrilateral base (`sorted`
4-tuple of vertices) plus four triangular faces (`sorted` 3-tuples).

Fields match [`Hex8FacetMaps`](@ref) layout counts for this topology:
`elem_edge_gid` `(8, nelem)`, `elem_face_gid` `(5, nelem)`.
"""
struct Pyr5FacetMaps <: AbstractFacetConnectivityMaps
    n_edges::Int
    n_faces::Int
    elem_edge_gid::Matrix{Int}
    elem_face_gid::Matrix{Int}
    elem_edge_orientation::Matrix{Int8}
    elem_face_orientation::Matrix{Int8}
    elem_edge_fraction::Matrix{Float64}
    elem_face_fraction::Matrix{Float64}
end

@inline function _pyr_sorted_face_triplet(conn::NTuple{5, UInt32}, verts::NTuple{3, Int})
    v = (conn[verts[1]], conn[verts[2]], conn[verts[3]])
    x = Int[v[1], v[2], v[3]]
    sort!(x)
    return (UInt32(x[1]), UInt32(x[2]), UInt32(x[3]))
end

@inline function _pyr_sorted_face_quad(conn::NTuple{5, UInt32}, verts::NTuple{4, Int})
    v = (conn[verts[1]], conn[verts[2]], conn[verts[3]], conn[verts[4]])
    x = Int[v[1], v[2], v[3], v[4]]
    sort!(x)
    return (UInt32(x[1]), UInt32(x[2]), UInt32(x[3]), UInt32(x[4]))
end

"""
    pyr_face_area_physical(X, local_face::Int) -> Float64

Face `local_face ∈ 1:5` on a linear pyramid: quad uses two triangles; triangles
use half the cross product norm (`X` length `5`).
"""
function pyr_face_area_physical(X::AbstractVector{V}, local_face::Int) where {V<:Vec{3}}
    fc = faces(Pyr5())[local_face]
    vs = fc.vertices
    n = length(vs)
    @inbounds p1 = X[vs[1]]
    @inbounds p2 = X[vs[2]]
    @inbounds p3 = X[vs[3]]
    if n == 3
        return 0.5 * norm((p2 - p1) × (p3 - p1))
    else
        @inbounds p4 = X[vs[4]]
        c1 = (p2 - p1) × (p3 - p1)
        c2 = (p3 - p1) × (p4 - p1)
        return 0.5 * (norm(c1) + norm(c2))
    end
end

"""
    pyr_edge_length_physical(X, local_edge::Int) -> Float64

Edge length for `edges(::Pyr5)` local index, `X` length `5`.
"""
function pyr_edge_length_physical(X::AbstractVector{V}, local_edge::Int) where {V<:Vec{3}}
    ed = edges(Pyr5())[local_edge]
    i, j = ed.vertices
    @inbounds return norm(X[i] - X[j])
end

"""
    pyr_face_outward_sign(X, local_face::Int) -> Int8

Outward normal convention consistent with [`hex8_face_outward_sign`](@ref) /
[`tet_face_outward_sign`](@ref).
"""
function pyr_face_outward_sign(X::AbstractVector{V}, local_face::Int) where {V<:Vec{3}}
    fc = faces(Pyr5())[local_face]
    vs = fc.vertices
    ce = _elem_centroid_mean(X)
    if length(vs) == 3
        @inbounds p1 = X[vs[1]]
        @inbounds p2 = X[vs[2]]
        @inbounds p3 = X[vs[3]]
        nvec = (p2 - p1) × (p3 - p1)
        cf = (p1 + p2 + p3) / 3
    else
        @inbounds p1 = X[vs[1]]
        @inbounds p2 = X[vs[2]]
        @inbounds p3 = X[vs[3]]
        @inbounds p4 = X[vs[4]]
        nvec = (p2 - p1) × (p3 - p1)
        cf = 0.25 * (p1 + p2 + p3 + p4)
    end
    s = dot(nvec, ce - cf)
    return s < 0 ? Int8(1) : Int8(-1)
end

@inline function _pyr5_element_coords(mesh::Mesh{5, Pyr5}, conn::NTuple{5, UInt32})
    return Vec{3, Float64}[mesh.nodes[Int(conn[i])] for i in 1:5]
end

"""
    build_pyr5_facet_maps(mesh::Mesh{5, Pyr5}) -> Pyr5FacetMaps
"""
function build_pyr5_facet_maps(mesh::Mesh{5, Pyr5})
    Ktop = Pyr5()
    edge_defs = edges(Ktop)
    face_defs = faces(Ktop)
    nelem = length(mesh.connectivity)

    edge_dict = Dict{Tuple{UInt32, UInt32}, Int}()
    tri_face_dict = Dict{NTuple{3, UInt32}, Int}()
    quad_face_dict = Dict{NTuple{4, UInt32}, Int}()

    elem_edge_gid = Matrix{Int}(undef, 8, nelem)
    elem_face_gid = Matrix{Int}(undef, 5, nelem)
    elem_edge_orientation = Matrix{Int8}(undef, 8, nelem)

    next_edge = 1
    next_face = 1

    @inbounds for eid in 1:nelem
        conn = mesh.connectivity[eid]

        for le in 1:8
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

        for lf in 1:5
            fc = face_defs[lf]
            fk = if length(fc.vertices) == 3
                _pyr_sorted_face_triplet(conn, fc.vertices)
            else
                _pyr_sorted_face_quad(conn, fc.vertices)
            end
            d = length(fc.vertices) == 3 ? tri_face_dict : quad_face_dict
            gid = get(d, fk, nothing)
            if gid === nothing
                gid = next_face
                d[fk] = gid
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
        for le in 1:8
            edge_touch[elem_edge_gid[le, eid]] += 1
        end
        for lf in 1:5
            face_touch[elem_face_gid[lf, eid]] += 1
        end
    end

    elem_edge_fraction = Matrix{Float64}(undef, 8, nelem)
    elem_face_fraction = Matrix{Float64}(undef, 5, nelem)
    elem_face_orientation = Matrix{Int8}(undef, 5, nelem)

    @inbounds for eid in 1:nelem
        conn = mesh.connectivity[eid]
        X = _pyr5_element_coords(mesh, conn)
        for le in 1:8
            g = elem_edge_gid[le, eid]
            elem_edge_fraction[le, eid] = 1.0 / edge_touch[g]
        end
        for lf in 1:5
            g = elem_face_gid[lf, eid]
            elem_face_fraction[lf, eid] = 1.0 / face_touch[g]
            elem_face_orientation[lf, eid] = pyr_face_outward_sign(X, lf)
        end
    end

    return Pyr5FacetMaps(
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
    pyr5_edge_orientation_sign(conn::NTuple{5,UInt32}, local_edge::Int) -> Int8

Directed-edge hint matching [`build_pyr5_facet_maps`](@ref).
"""
function pyr5_edge_orientation_sign(conn::NTuple{5, UInt32}, local_edge::Int)
    ed = edges(Pyr5())[local_edge]
    va, vb = ed.vertices
    na = Int(conn[va])
    nb = Int(conn[vb])
    return na < nb ? Int8(1) : Int8(-1)
end
