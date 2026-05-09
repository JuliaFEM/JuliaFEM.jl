# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using Test
using LinearAlgebra
using JuliaFEM
using JuliaFEM: FacetMassKernel, EdgeMassKernel
using JuliaFEM: DOFBasedCOOAssembler, DOFBasedCOOCache, assemble!, extract_system

# Reference Tet4 from `test_tet4_facet_maps.jl`.
function _reference_tet4_mesh()
    nodes = Vec{3, Float64}[
        Vec(0.0, 0.0, 0.0),
        Vec(1.0, 0.0, 0.0),
        Vec(0.0, 1.0, 0.0),
        Vec(0.0, 0.0, 1.0),
    ]
    conn = (UInt32(1), UInt32(2), UInt32(3), UInt32(4))
    return Mesh{Tet4}(nodes, [conn]), nodes
end

@testset "DOFHandler Vec{2} on Edge — two globals per topological edge" begin
    mesh, _ = _reference_tet4_mesh()
    S = @DOFSet{circ::DOF{Vec{2, Float64}, Edge}}
    elements, handler = create_elements!(mesh, Element{Tet4, Lagrange{1}, S})
    @test handler.total_dofs == 12
    @test length(elements) == 1
    @test length(elements[1].dof_indices) == 12
end

@testset "EdgeMassKernel diagonal — Vec{2} replicates scalar edge measure per component" begin
    mesh, nodes = _reference_tet4_mesh()
    X = nodes
    expected_edge_len = sum(tet_edge_length_physical(X, le) for le in 1:6)

    S = @DOFSet{circ::DOF{Vec{2, Float64}, Edge}}
    elements, handler = create_elements!(mesh, Element{Tet4, Lagrange{1}, S})
    kernel = EdgeMassKernel(mesh)
    cache = DOFBasedCOOCache(elements, handler, mesh, kernel)
    assemble!(cache, DOFBasedCOOAssembler(), kernel, mesh)
    K, _ = extract_system(cache)
    @test sum(diag(Matrix(K))) ≈ 2 * expected_edge_len rtol = 1e-10
end

@testset "FacetMassKernel diagonal — Vec{2} on Face (Hex8 unit cube)" begin
    mesh = create_unit_cube_mesh(Hex8; nx=1, ny=1, nz=1)
    @test nelements(mesh) == 1
    conn = mesh.connectivity[1]
    X = Vec{3, Float64}[mesh.nodes[Int(conn[i])] for i in 1:8]
    expected_face_area = sum(hex8_face_area_physical(X, lf) for lf in 1:6)
    @test expected_face_area ≈ 6.0 rtol = 1e-12

    S = @DOFSet{flux::DOF{Vec{2, Float64}, Face}}
    elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    kernel = FacetMassKernel(mesh)
    cache = DOFBasedCOOCache(elements, handler, mesh, kernel)
    assemble!(cache, DOFBasedCOOAssembler(), kernel, mesh)
    K, _ = extract_system(cache)
    @test sum(diag(Matrix(K))) ≈ 2 * expected_face_area rtol = 1e-10
end

@testset "FacetMassKernel diagonal — Vec{2} on Face (Tet4)" begin
    mesh, nodes = _reference_tet4_mesh()
    X = nodes
    expected_face_area = sum(tet_face_area_physical(X, lf) for lf in 1:4)

    S = @DOFSet{flux::DOF{Vec{2, Float64}, Face}}
    elements, handler = create_elements!(mesh, Element{Tet4, Lagrange{1}, S})
    kernel = FacetMassKernel(mesh)
    cache = DOFBasedCOOCache(elements, handler, mesh, kernel)
    assemble!(cache, DOFBasedCOOAssembler(), kernel, mesh)
    K, _ = extract_system(cache)
    @test sum(diag(Matrix(K))) ≈ 2 * expected_face_area rtol = 1e-10
end
