# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using Test
using JuliaFEM
using JuliaFEM: Tet4FacetMaps, build_tet4_facet_maps, tet4_edge_orientation_sign
using JuliaFEM: FacetMassKernel, EdgeMassKernel
using JuliaFEM: DOFBasedCOOAssembler, DOFBasedCOOCache, assemble!, extract_system
using LinearAlgebra

@testset "build_tet4_facet_maps (single tet)" begin
    nodes = Vec{3, Float64}[
        Vec(0.0, 0.0, 0.0),
        Vec(1.0, 0.0, 0.0),
        Vec(0.0, 1.0, 0.0),
        Vec(0.0, 0.0, 1.0),
    ]
    conn = (UInt32(1), UInt32(2), UInt32(3), UInt32(4))
    mesh = Mesh{Tet4}(nodes, [conn])
    maps = build_tet4_facet_maps(mesh)
    @test maps isa Tet4FacetMaps
    @test maps.n_edges == 6
    @test maps.n_faces == 4
    @test size(maps.elem_face_orientation) == (4, 1)
    @test all(o -> abs(o) == 1, maps.elem_face_orientation)

    for le in 1:6
        @test tet4_edge_orientation_sign(conn, le) == maps.elem_edge_orientation[le, 1]
    end

    S = @DOFSet{e::DOF{Float64, Edge}}
    elements, handler = create_elements!(mesh, Element{Tet4, Lagrange{1}, S})
    @test handler.total_dofs == 6
end

@testset "two Tet4 sharing one triangular face" begin
    nodes = Vec{3, Float64}[
        Vec(0.0, 0.0, 0.0),
        Vec(1.0, 0.0, 0.0),
        Vec(0.0, 1.0, 0.0),
        Vec(0.0, 0.0, 1.0),
        Vec(0.0, 0.0, -1.0),
    ]
    c1 = (UInt32(1), UInt32(2), UInt32(3), UInt32(4))
    c2 = (UInt32(1), UInt32(2), UInt32(3), UInt32(5))
    mesh = Mesh{Tet4}(nodes, [c1, c2])
    maps = build_tet4_facet_maps(mesh)
    @test maps.n_faces == 7
    @test maps.n_edges == 9

    products = Int[]
    for le1 in 1:6, le2 in 1:6
        if maps.elem_edge_gid[le1, 1] == maps.elem_edge_gid[le2, 2]
            push!(products, Int(maps.elem_edge_orientation[le1, 1] * maps.elem_edge_orientation[le2, 2]))
        end
    end
    @test !isempty(products)
    @test all(abs(p) == 1 for p in products)
    # Same nodal edge-direction convention as Hex8 — shared edges can match (+1,+1).
    @test any(==(1), products)
end

@testset "FacetMassKernel / EdgeMassKernel on reference Tet4" begin
    nodes = Vec{3, Float64}[
        Vec(0.0, 0.0, 0.0),
        Vec(1.0, 0.0, 0.0),
        Vec(0.0, 1.0, 0.0),
        Vec(0.0, 0.0, 1.0),
    ]
    conn = (UInt32(1), UInt32(2), UInt32(3), UInt32(4))
    mesh = Mesh{Tet4}(nodes, [conn])

    Sf = @DOFSet{flux::DOF{Float64, Face}}
    ef, hf = create_elements!(mesh, Element{Tet4, Lagrange{1}, Sf})
    kf = FacetMassKernel(mesh)
    cf = DOFBasedCOOCache(ef, hf, mesh, kf)
    assemble!(cf, DOFBasedCOOAssembler(), kf, mesh)
    Kf, _ = extract_system(cf)
    d = diag(Matrix(Kf))
    expected_faces = 1.5 + sqrt(3) / 2  # three right triangles + equilateral √2-side face
    @test sum(d) ≈ expected_faces rtol = 1e-10

    Se = @DOFSet{circ::DOF{Float64, Edge}}
    ee, he = create_elements!(mesh, Element{Tet4, Lagrange{1}, Se})
    ke = EdgeMassKernel(mesh)
    ce = DOFBasedCOOCache(ee, he, mesh, ke)
    assemble!(ce, DOFBasedCOOAssembler(), ke, mesh)
    Ke, _ = extract_system(ce)
    de = diag(Matrix(Ke))
    expected_edges = 3.0 + 3 * sqrt(2.0)
    @test sum(de) ≈ expected_edges rtol = 1e-10
end
