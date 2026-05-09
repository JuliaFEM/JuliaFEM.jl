# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using Test
using JuliaFEM
using JuliaFEM: Wedge6FacetMaps, build_wedge6_facet_maps, wedge6_edge_orientation_sign
using JuliaFEM: FacetMassKernel, EdgeMassKernel
using JuliaFEM: DOFBasedCOOAssembler, DOFBasedCOOCache, assemble!, extract_system

@testset "build_wedge6_facet_maps (single wedge)" begin
    nodes = Vec{3, Float64}[reference_coordinates(Wedge6())...]
    conn = ntuple(i -> UInt32(i), 6)
    mesh = Mesh{Wedge6}(nodes, [conn])
    maps = build_wedge6_facet_maps(mesh)
    @test maps isa Wedge6FacetMaps
    @test maps.n_edges == 9
    @test maps.n_faces == 5
    @test size(maps.elem_face_orientation) == (5, 1)
    @test all(o -> abs(o) == 1, maps.elem_face_orientation)

    for le in 1:9
        @test wedge6_edge_orientation_sign(conn, le) == maps.elem_edge_orientation[le, 1]
    end

    S = @DOFSet{e::DOF{Float64, Edge}}
    elements, handler = create_elements!(mesh, Element{Wedge6, Lagrange{1}, S})
    @test handler.total_dofs == 9
    @test elements[1].dof_indices isa NTuple{9, UInt64}
end

@testset "FacetMassKernel / EdgeMassKernel on reference Wedge6" begin
    nodes = Vec{3, Float64}[reference_coordinates(Wedge6())...]
    conn = ntuple(i -> UInt32(i), 6)
    mesh = Mesh{Wedge6}(nodes, [conn])
    X = nodes

    expected_face_area = sum(wedge_face_area_physical(X, lf) for lf in 1:5)
    expected_edge_len = sum(wedge_edge_length_physical(X, le) for le in 1:9)

    Sf = @DOFSet{flux::DOF{Float64, Face}}
    ef, hf = create_elements!(mesh, Element{Wedge6, Lagrange{1}, Sf})
    kf = FacetMassKernel(mesh)
    cf = DOFBasedCOOCache(ef, hf, mesh, kf)
    assemble!(cf, DOFBasedCOOAssembler(), kf, mesh)
    Kf, _ = extract_system(cf)
    @test sum(diag(Matrix(Kf))) ≈ expected_face_area rtol = 1e-10

    Se = @DOFSet{circ::DOF{Float64, Edge}}
    ee, he = create_elements!(mesh, Element{Wedge6, Lagrange{1}, Se})
    ke = EdgeMassKernel(mesh)
    ce = DOFBasedCOOCache(ee, he, mesh, ke)
    assemble!(ce, DOFBasedCOOAssembler(), ke, mesh)
    Ke, _ = extract_system(ce)
    @test sum(diag(Matrix(Ke))) ≈ expected_edge_len rtol = 1e-10
end
