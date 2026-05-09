# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using Test
using JuliaFEM
using JuliaFEM: Hex8FacetMaps, build_hex8_facet_maps, FacetMassKernel, EdgeMassKernel
using JuliaFEM: hex8_edge_length_physical, hex8_edge_orientation_sign
using JuliaFEM: field_ndofs
using JuliaFEM: DOFBasedCOOAssembler, DOFBasedCOOCache, assemble!, extract_system
using LinearAlgebra

@testset "build_hex8_facet_maps (single Hex8)" begin
    mesh = create_unit_cube_mesh(Hex8)
    maps = build_hex8_facet_maps(mesh)
    @test maps.n_edges == 12
    @test maps.n_faces == 6
    @test size(maps.elem_face_gid) == (6, 1)
    @test size(maps.elem_edge_orientation) == (12, 1)
    @test size(maps.elem_face_orientation) == (6, 1)
    @test all(o -> abs(o) == 1, maps.elem_face_orientation)
    @test all(==(1), maps.elem_face_fraction)
end

@testset "hex8_edge_orientation_sign vs facet maps" begin
    mesh = create_unit_cube_mesh(Hex8)
    maps = build_hex8_facet_maps(mesh)
    conn = mesh.connectivity[1]
    for le in 1:12
        @test hex8_edge_orientation_sign(conn, le) == maps.elem_edge_orientation[le, 1]
    end
end

@testset "edge orientation on shared hex–hex edges (2×1×1 bar)" begin
    mesh = create_structured_box_mesh(Hex8; xmin = 0.0, xmax = 2.0, nx = 2, ny = 1, nz = 1)
    maps = build_hex8_facet_maps(mesh)
    products = Int[]
    for le1 in 1:12, le2 in 1:12
        if maps.elem_edge_gid[le1, 1] == maps.elem_edge_gid[le2, 2]
            push!(products, Int(maps.elem_edge_orientation[le1, 1] * maps.elem_edge_orientation[le2, 2]))
        end
    end
    @test !isempty(products)
    @test all(abs(p) == 1 for p in products)
    # Some shared edges see opposite local directions (−1); others agree (+1)
    # — both occur on this structured mesh.
    @test any(==(-1), products)
    @test any(==(1), products)
end

@testset "RT0FaceFlux / Nedelec1Edge dof counts (tags vs Float64)" begin
    @test field_ndofs(DOF{Nedelec1Edge, Edge}, Hex8) == field_ndofs(DOF{Float64, Edge}, Hex8)
    @test field_ndofs(DOF{RT0FaceFlux, Face}, Hex8) == field_ndofs(DOF{Float64, Face}, Hex8)
end

@testset "DOFHandler Face field + FacetMassKernel (unit cube)" begin
    mesh = create_unit_cube_mesh(Hex8)
    S = @DOFSet{flux::DOF{Float64, Face}}
    elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    @test handler.facet_maps isa Hex8FacetMaps
    @test handler.total_dofs == 6

    kernel = FacetMassKernel(mesh)
    asm = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elements, handler, mesh, kernel)
    assemble!(cache, asm, kernel, mesh)
    K, _ = extract_system(cache)
    @test size(K) == (6, 6)

    d = diag(Matrix(K))
    @test all(d .> 0)
    @test sum(d) ≈ 6.0 rtol = 1e-10
    off = K - Diagonal(d)
    @test norm(off) < 1e-12
end

@testset "FacetMassKernel two-element bar (shared interior face)" begin
    mesh = create_structured_box_mesh(Hex8; xmin = 0.0, xmax = 2.0, nx = 2, ny = 1, nz = 1)
    @test length(mesh.connectivity) == 2
    S = @DOFSet{flux::DOF{Float64, Face}}
    elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    # Two hexes share one interior facet: 12 − 1 merged pair => 11 unique mesh faces.
    @test handler.total_dofs == 11

    kernel = FacetMassKernel(mesh)
    asm = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elements, handler, mesh, kernel)
    assemble!(cache, asm, kernel, mesh)
    K, _ = extract_system(cache)
    d = diag(Matrix(K))
    # Sum of face areas over all mesh facets (exterior SA is 10; interior shared quad adds 1).
    @test sum(d) ≈ 11.0 rtol = 1e-10
end

@testset "hex8_edge_length_physical (unit Hex8)" begin
    mesh = create_unit_cube_mesh(Hex8)
    conn = mesh.connectivity[1]
    X = Vec{3,Float64}[mesh.nodes[Int(conn[i])] for i in 1:8]
    @test hex8_edge_length_physical(X, 1) ≈ 1.0
    @test hex8_edge_length_physical(X, 9) ≈ 1.0
end

@testset "DOFHandler Edge field + EdgeMassKernel (unit cube)" begin
    mesh = create_unit_cube_mesh(Hex8)
    S = @DOFSet{circ::DOF{Float64, Edge}}
    elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    @test handler.facet_maps isa Hex8FacetMaps
    @test handler.total_dofs == 12

    kernel = EdgeMassKernel(mesh)
    asm = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elements, handler, mesh, kernel)
    assemble!(cache, asm, kernel, mesh)
    K, _ = extract_system(cache)
    @test size(K) == (12, 12)

    d = diag(Matrix(K))
    @test all(d .> 0)
    @test sum(d) ≈ 12.0 rtol = 1e-10
    off = K - Diagonal(d)
    @test norm(off) < 1e-12
end

@testset "EdgeMassKernel two-element bar (shared interior face)" begin
    mesh = create_structured_box_mesh(Hex8; xmin = 0.0, xmax = 2.0, nx = 2, ny = 1, nz = 1)
    S = @DOFSet{circ::DOF{Float64, Edge}}
    elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    # Two hexes share one interior quad facet ⇒ its four bounding edges merge pairwise:
    # 24 − 4 = 20 unique mesh edges (see unique-edge counting on conforming meshes).
    @test handler.total_dofs == 20

    kernel = EdgeMassKernel(mesh)
    asm = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elements, handler, mesh, kernel)
    assemble!(cache, asm, kernel, mesh)
    K, _ = extract_system(cache)
    d = diag(Matrix(K))
    @test sum(d) ≈ 20.0 rtol = 1e-10
end
