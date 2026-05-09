# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using Test
using LinearAlgebra
using JuliaFEM
using JuliaFEM: Hex8FacetMaps, build_hex20_facet_maps, hex20_edge_orientation_sign
using JuliaFEM: FacetMassKernel, EdgeMassKernel
using JuliaFEM: DOFBasedCOOAssembler, DOFBasedCOOCache, assemble!, extract_system

@testset "build_hex20_facet_maps (single Hex20)" begin
    nodes = Vec{3, Float64}[reference_coordinates(Hex20())...]
    conn = ntuple(i -> UInt32(i), 20)
    mesh = Mesh{Hex20}(nodes, [conn])
    maps = build_hex20_facet_maps(mesh)
    @test maps isa Hex8FacetMaps
    @test maps.n_edges == 12
    @test maps.n_faces == 6

    for le in 1:12
        @test hex20_edge_orientation_sign(conn, le) == maps.elem_edge_orientation[le, 1]
    end

    S = @DOFSet{e::DOF{Float64, Edge}}
    elements, handler = create_elements!(mesh, Element{Hex20, Serendipity{2}, S})
    @test handler.total_dofs == 12
end

@testset "FacetMassKernel / EdgeMassKernel on reference Hex20 (corner skeleton)" begin
    nodes = Vec{3, Float64}[reference_coordinates(Hex20())...]
    conn = ntuple(i -> UInt32(i), 20)
    mesh = Mesh{Hex20}(nodes, [conn])
    X = nodes

    expected_face_area = sum(hex8_face_area_physical(X, lf) for lf in 1:6)
    expected_edge_len = sum(hex8_edge_length_physical(X, le) for le in 1:12)

    Sf = @DOFSet{flux::DOF{Float64, Face}}
    ef, hf = create_elements!(mesh, Element{Hex20, Serendipity{2}, Sf})
    kf = FacetMassKernel(mesh)
    cf = DOFBasedCOOCache(ef, hf, mesh, kf)
    assemble!(cf, DOFBasedCOOAssembler(), kf, mesh)
    Kf, _ = extract_system(cf)
    @test sum(diag(Matrix(Kf))) ≈ expected_face_area rtol = 1e-10

    Se = @DOFSet{circ::DOF{Float64, Edge}}
    ee, he = create_elements!(mesh, Element{Hex20, Serendipity{2}, Se})
    ke = EdgeMassKernel(mesh)
    ce = DOFBasedCOOCache(ee, he, mesh, ke)
    assemble!(ce, DOFBasedCOOAssembler(), ke, mesh)
    Ke, _ = extract_system(ce)
    @test sum(diag(Matrix(Ke))) ≈ expected_edge_len rtol = 1e-10
end
