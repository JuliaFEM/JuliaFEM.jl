# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using Test
using LinearAlgebra
using JuliaFEM
using JuliaFEM: Pyr5FacetMaps, build_pyr5_facet_maps, pyr5_edge_orientation_sign
using JuliaFEM: FacetMassKernel, EdgeMassKernel
using JuliaFEM: DOFBasedCOOAssembler, DOFBasedCOOCache, assemble!, extract_system

@testset "build_pyr5_facet_maps (single pyramid)" begin
    nodes = Vec{3, Float64}[reference_coordinates(Pyr5())...]
    conn = ntuple(i -> UInt32(i), 5)
    mesh = Mesh{Pyr5}(nodes, [conn])
    maps = build_pyr5_facet_maps(mesh)
    @test maps isa Pyr5FacetMaps
    @test maps.n_edges == 8
    @test maps.n_faces == 5

    for le in 1:8
        @test pyr5_edge_orientation_sign(conn, le) == maps.elem_edge_orientation[le, 1]
    end

    S = @DOFSet{e::DOF{Float64, Edge}}
    elements, handler = create_elements!(mesh, Element{Pyr5, Lagrange{1}, S})
    @test handler.total_dofs == 8
    @test elements[1].dof_indices isa NTuple{8, UInt64}
end

@testset "FacetMassKernel / EdgeMassKernel on reference Pyr5" begin
    nodes = Vec{3, Float64}[reference_coordinates(Pyr5())...]
    conn = ntuple(i -> UInt32(i), 5)
    mesh = Mesh{Pyr5}(nodes, [conn])
    X = nodes

    expected_face_area = sum(pyr_face_area_physical(X, lf) for lf in 1:5)
    expected_edge_len = sum(pyr_edge_length_physical(X, le) for le in 1:8)

    Sf = @DOFSet{flux::DOF{Float64, Face}}
    ef, hf = create_elements!(mesh, Element{Pyr5, Lagrange{1}, Sf})
    kf = FacetMassKernel(mesh)
    cf = DOFBasedCOOCache(ef, hf, mesh, kf)
    assemble!(cf, DOFBasedCOOAssembler(), kf, mesh)
    Kf, _ = extract_system(cf)
    @test sum(diag(Matrix(Kf))) ≈ expected_face_area rtol = 1e-10

    Se = @DOFSet{circ::DOF{Float64, Edge}}
    ee, he = create_elements!(mesh, Element{Pyr5, Lagrange{1}, Se})
    ke = EdgeMassKernel(mesh)
    ce = DOFBasedCOOCache(ee, he, mesh, ke)
    assemble!(ce, DOFBasedCOOAssembler(), ke, mesh)
    Ke, _ = extract_system(ce)
    @test sum(diag(Matrix(Ke))) ≈ expected_edge_len rtol = 1e-10
end
