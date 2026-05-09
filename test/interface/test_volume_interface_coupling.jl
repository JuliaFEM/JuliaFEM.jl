# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using LinearAlgebra: norm
using Test

"""Return local vertex index `i` with `conn[i] == g` (1-based)."""
function _hex8_local_vertex(conn::NTuple{8, UInt32}, g::UInt32)
    for i in 1:8
        @inbounds conn[i] == g && return i
    end
    error("global node $g not found in connectivity $conn")
end

"""Reference-coordinate midpoint of the linear hex edge joining two vertices."""
function _hex8_edge_midpoint_ref(conn::NTuple{8, UInt32}, g1::UInt32, g2::UInt32)
    R = reference_coordinates(Hex8())
    i1 = _hex8_local_vertex(conn, g1)
    i2 = _hex8_local_vertex(conn, g2)
    return Vec{3, Float64}(0.5 * (R[i1] + R[i2]))
end

"""Global node ids on hex face `lf` (1-based), cyclic order from topology."""
function _hex8_face_global_nodes(conn::NTuple{8, UInt32}, lf::Int)
    F = faces(Hex8())[lf]
    return ntuple(i -> conn[F.vertices[i]], length(F.vertices))
end

"""Two-element brick sharing an interior face (nx = 2)."""
function _two_hex_brick_mesh()
    return create_structured_box_mesh(Hex8;
        xmin = 0.0, xmax = 1.0, nx = 2,
        ymin = 0.0, ymax = 1.0, ny = 1,
        zmin = 0.0, zmax = 1.0, nz = 1,
    )
end

"""Find local face indices on elements 1 and 2 that share the same global mesh face."""
function _shared_hex8_face_pair(maps::Hex8FacetMaps)
    lf1 = lf2 = 0
    for a in 1:6, b in 1:6
        if maps.elem_face_gid[a, 1] == maps.elem_face_gid[b, 2]
            lf1, lf2 = a, b
            break
        end
    end
    lf1 == 0 && error("expected conforming shared face between elements 1 and 2")
    return lf1, lf2
end

@testset "volume–interface coupling (1) geometry & facet topology" begin
    mesh = _two_hex_brick_mesh()
    @test nelements(mesh) == 2

    maps = build_hex8_facet_maps(mesh)
    lf1, lf2 = _shared_hex8_face_pair(maps)
    @test maps.elem_face_gid[lf1, 1] == maps.elem_face_gid[lf2, 2]

    conn1 = mesh.connectivity[1]
    conn2 = mesh.connectivity[2]
    face1 = Set(collect(_hex8_face_global_nodes(conn1, lf1)))
    face2 = Set(collect(_hex8_face_global_nodes(conn2, lf2)))
    @test face1 == face2

    face_cycle = collect(_hex8_face_global_nodes(conn1, lf1))
    nfv = length(face_cycle)
    seg_pairs = Tuple{UInt32, UInt32}[]
    for k in 1:nfv
        a = face_cycle[k]
        b = face_cycle[mod1(k + 1, nfv)]
        push!(seg_pairs, (UInt32(a), UInt32(b)))
    end

    iface_nodes = [mesh.nodes[Int(i)] for i in face_cycle]
    iface_conn = [(UInt32(k), UInt32(mod1(k + 1, nfv))) for k in 1:nfv]
    coup = [
        InterfaceVolumeCoupling(UInt32(1), UInt32(1), UInt8(lf1), UInt32(1), UInt32(2), UInt8(lf2))
        for _ in 1:nfv
    ]
    im = InterfaceMesh(Seg2, iface_nodes, iface_conn, coup)

    vol_gid = UInt32.(face_cycle)
    for i in 1:interface_nnodes(im)
        @test im.nodes[i] ≈ mesh.nodes[Int(vol_gid[i])]
    end
    for (s, (ga, gb)) in enumerate(seg_pairs)
        @test ga ∈ face1 && gb ∈ face1
        la = findfirst(==(ga), vol_gid)
        lb = findfirst(==(gb), vol_gid)
        @test im.nodes[la] ≈ mesh.nodes[Int(ga)]
        @test im.nodes[lb] ≈ mesh.nodes[Int(gb)]
        c = im.connectivity[s]
        @test sort(UInt32[vol_gid[Int(c[1])], vol_gid[Int(c[2])]]) == sort(UInt32[ga, gb])
    end
end

@testset "volume–interface coupling (2) scalar restriction along shared face" begin
    mesh = _two_hex_brick_mesh()
    maps = build_hex8_facet_maps(mesh)
    lf1, lf2 = _shared_hex8_face_pair(maps)

    conn1 = mesh.connectivity[1]
    face_cycle = collect(_hex8_face_global_nodes(conn1, lf1))
    nfv = length(face_cycle)

    iface_nodes = [mesh.nodes[Int(i)] for i in face_cycle]
    iface_conn = [(UInt32(k), UInt32(mod1(k + 1, nfv))) for k in 1:nfv]
    coup = [
        InterfaceVolumeCoupling(UInt32(1), UInt32(1), UInt8(lf1), UInt32(1), UInt32(2), UInt8(lf2))
        for _ in 1:nfv
    ]
    im = InterfaceMesh(Seg2, iface_nodes, iface_conn, coup)

    S = @DOFSet{T::DOF{Float64, Vertex}}
    vol_elements, vol_h = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    _, iface_h = create_interface_elements!(im, Element{Seg2, Lagrange{1}, S})

    u_vol = zeros(Float64, vol_h.total_dofs)
    for n in 1:nnodes_total(mesh)
        g = vol_h.field_starts[1][n]
        u_vol[g] = mesh.nodes[n][1]
    end

    vol_gid = UInt32.(face_cycle)
    u_iface = zeros(Float64, iface_h.total_dofs)
    for i in 1:interface_nnodes(im)
        v = vol_gid[i]
        u_iface[iface_h.field_starts[1][i]] = u_vol[vol_h.field_starts[1][Int(v)]]
    end
    for i in 1:interface_nnodes(im)
        @test u_iface[iface_h.field_starts[1][i]] ≈ im.nodes[i][1]
    end
end

@testset "volume–interface coupling (3) segment midpoint jump & mortar-style residual" begin
    mesh = _two_hex_brick_mesh()
    maps = build_hex8_facet_maps(mesh)
    lf1, lf2 = _shared_hex8_face_pair(maps)

    conn1 = mesh.connectivity[1]
    conn2 = mesh.connectivity[2]
    face_cycle = collect(_hex8_face_global_nodes(conn1, lf1))
    nfv = length(face_cycle)

    iface_nodes = [mesh.nodes[Int(i)] for i in face_cycle]
    iface_conn = [(UInt32(k), UInt32(mod1(k + 1, nfv))) for k in 1:nfv]
    coup = [
        InterfaceVolumeCoupling(UInt32(1), UInt32(1), UInt8(lf1), UInt32(1), UInt32(2), UInt8(lf2))
        for _ in 1:nfv
    ]
    im = InterfaceMesh(Seg2, iface_nodes, iface_conn, coup)

    S = @DOFSet{T::DOF{Float64, Vertex}}
    vol_elements, vol_h = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})

    u_vol = zeros(Float64, vol_h.total_dofs)
    for n in 1:nnodes_total(mesh)
        u_vol[vol_h.field_starts[1][n]] = mesh.nodes[n][1]
    end

    penalty = 0.0
    mortar_sum = 0.0
    for s in 1:interface_nelements(im)
        g1 = UInt32(face_cycle[s])
        g2 = UInt32(face_cycle[mod1(s + 1, nfv)])
        ξ1 = _hex8_edge_midpoint_ref(conn1, g1, g2)
        ξ2 = _hex8_edge_midpoint_ref(conn2, g1, g2)
        u_slave = interpolate_field_value(vol_elements[1], u_vol, :T, ξ1)
        u_master = interpolate_field_value(vol_elements[2], u_vol, :T, ξ2)
        @test u_slave ≈ u_master

        L = norm(im.nodes[s] - im.nodes[mod1(s + 1, nfv)])
        umid_line = 0.5 * (u_vol[vol_h.field_starts[1][Int(g1)]] + u_vol[vol_h.field_starts[1][Int(g2)]])
        @test umid_line ≈ u_slave

        penalty += (u_slave - u_master)^2 * L
        λ = 1.0
        mortar_sum += λ * L * (u_slave - u_master)
    end
    @test penalty ≈ 0.0 atol = 1e-13 rtol = 1e-13
    @test mortar_sum ≈ 0.0 atol = 1e-13 rtol = 1e-13
end
