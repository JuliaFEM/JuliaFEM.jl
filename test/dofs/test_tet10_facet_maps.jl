# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using Test
using LinearAlgebra
using JuliaFEM
using JuliaFEM: Tet4FacetMaps, build_tet10_facet_maps, tet10_edge_orientation_sign
using JuliaFEM: FacetMassKernel, EdgeMassKernel
using JuliaFEM: DOFBasedCOOAssembler, DOFBasedCOOCache, assemble!, extract_system

@testset "build_tet10_facet_maps (single Tet10)" begin
    nodes = Vec{3, Float64}[reference_coordinates(Tet10())...]
    conn = ntuple(i -> UInt32(i), 10)
    mesh = Mesh{Tet10}(nodes, [conn])
    maps = build_tet10_facet_maps(mesh)
    @test maps isa Tet4FacetMaps
    @test maps.n_edges == 6
    @test maps.n_faces == 4

    for le in 1:6
        @test tet10_edge_orientation_sign(conn, le) == maps.elem_edge_orientation[le, 1]
    end

    S = @DOFSet{e::DOF{Float64, Edge}}
    elements, handler = create_elements!(mesh, Element{Tet10, Lagrange{2}, S})
    @test handler.total_dofs == 6
end

@testset "FacetMassKernel / EdgeMassKernel on reference Tet10 (corner skeleton)" begin
    nodes = Vec{3, Float64}[reference_coordinates(Tet10())...]
    conn = ntuple(i -> UInt32(i), 10)
    mesh = Mesh{Tet10}(nodes, [conn])
    X = nodes

    expected_face_area = sum(tet_face_area_physical(X, lf) for lf in 1:4)
    expected_edge_len = sum(tet_edge_length_physical(X, le) for le in 1:6)

    Sf = @DOFSet{flux::DOF{Float64, Face}}
    ef, hf = create_elements!(mesh, Element{Tet10, Lagrange{2}, Sf})
    kf = FacetMassKernel(mesh)
    cf = DOFBasedCOOCache(ef, hf, mesh, kf)
    assemble!(cf, DOFBasedCOOAssembler(), kf, mesh)
    Kf, _ = extract_system(cf)
    @test sum(diag(Matrix(Kf))) ≈ expected_face_area rtol = 1e-10

    Se = @DOFSet{circ::DOF{Float64, Edge}}
    ee, he = create_elements!(mesh, Element{Tet10, Lagrange{2}, Se})
    ke = EdgeMassKernel(mesh)
    ce = DOFBasedCOOCache(ee, he, mesh, ke)
    assemble!(ce, DOFBasedCOOAssembler(), ke, mesh)
    Ke, _ = extract_system(ce)
    @test sum(diag(Matrix(Ke))) ≈ expected_edge_len rtol = 1e-10
end
