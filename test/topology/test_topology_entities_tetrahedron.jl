# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using Test
using StaticArrays

@testset "Topology Entities: Tetrahedron" begin
    
    @testset "Basic topology properties" begin
        topo = Tet4()
        @test nnodes(topo) == 4
        @test dim(topo) == 3
        @test nvertices(topo) == 4
        @test nedges(topo) == 6
        @test nfaces(topo) == 4
    end
    
    @testset "Vertex entities" begin
        topo = Tet4()
        verts = vertices(topo)
        
        @test verts isa SVector{4, Vertex}
        @test length(verts) == 4
        @test eltype(verts) == Vertex
        
        # Test count helper
        @test nentities(Tetrahedron{4}, Vertex) == 4
    end
    
    @testset "Edge entities" begin
        topo = Tet4()
        edges_list = edges(topo)
        
        @test edges_list isa SVector{6, Edge}
        @test length(edges_list) == 6
        @test eltype(edges_list) == Edge
        
        # Check edge connectivity
        @test edges_list[1].vertices == (1, 2)
        @test edges_list[2].vertices == (2, 3)
        @test edges_list[3].vertices == (3, 1)
        @test edges_list[4].vertices == (1, 4)
        @test edges_list[5].vertices == (2, 4)
        @test edges_list[6].vertices == (3, 4)
        
        # Test count helper
        @test nentities(Tetrahedron{4}, Edge) == 6
    end
    
    @testset "Face entities" begin
        topo = Tet4()
        faces_list = faces(topo)
        
        @test faces_list isa SVector{4, Face}
        @test length(faces_list) == 4
        @test eltype(faces_list) == Face
        
        # Check face connectivity
        @test faces_list[1].vertices == (1, 3, 2)
        @test faces_list[2].vertices == (1, 2, 4)
        @test faces_list[3].vertices == (2, 3, 4)
        @test faces_list[4].vertices == (3, 1, 4)
        
        # Test count helper
        @test nentities(Tetrahedron{4}, Face) == 4
    end
    
    @testset "Cell entities" begin
        topo = Tet4()
        cells_list = cells(topo)
        
        @test cells_list isa SVector{1, Cell}
        @test length(cells_list) == 1
        @test eltype(cells_list) == Cell
        
        # Test count helper
        @test nentities(Tetrahedron{4}, Cell) == 1
    end
    
    @testset "Entity dimension queries" begin
        @test dim(Vertex) == 0
        @test dim(Edge) == 1
        @test dim(Face) == 2
        @test dim(Cell) == 3
    end
    
    @testset "Topological invariance: Tet4 vs Tet10" begin
        topo4 = Tet4()
        topo10 = Tet10()
        
        # Node counts differ (4 vs 10)
        @test nnodes(topo4) == 4
        @test nnodes(topo10) == 10
        
        # But topological structure is identical
        @test nvertices(topo4) == nvertices(topo10)  # Both 4
        @test nedges(topo4) == nedges(topo10)        # Both 6
        @test nfaces(topo4) == nfaces(topo10)        # Both 4
        
        # Edge connectivity is identical
        edges4 = edges(topo4)
        edges10 = edges(topo10)
        @test edges4[1].vertices == edges10[1].vertices  # (1, 2)
        @test edges4[3].vertices == edges10[3].vertices  # (3, 1)
        @test edges4[6].vertices == edges10[6].vertices  # (3, 4)
        
        # Face connectivity is identical
        faces4 = faces(topo4)
        faces10 = faces(topo10)
        @test faces4[1].vertices == faces10[1].vertices  # (1, 3, 2)
        @test faces4[4].vertices == faces10[4].vertices  # (3, 1, 4)
    end
    
    @testset "DOF system integration concept" begin
        # This tests the design pattern, not actual DOF implementation
        topo = Tet4()
        
        # Simulate Nedelec edge element: Vec{3} DOF on each edge
        nedges_dof = nentities(Tetrahedron{4}, Edge)
        components_per_edge = 3  # Vec{3}
        total_dofs = nedges_dof * components_per_edge
        @test total_dofs == 18  # 6 edges × 3 components
        
        # Simulate Raviart-Thomas face element: Vec{3} DOF on each face
        nfaces_dof = nentities(Tetrahedron{4}, Face)
        components_per_face = 3  # Vec{3}
        total_dofs = nfaces_dof * components_per_face
        @test total_dofs == 12  # 4 faces × 3 components
        
        # Simulate Lagrange vertex element: Float64 DOF on each vertex
        nvertices_dof = nentities(Tetrahedron{4}, Vertex)
        components_per_vertex = 1  # Float64
        total_dofs = nvertices_dof * components_per_vertex
        @test total_dofs == 4  # 4 vertices × 1 component
    end
    
    @testset "Entity position as ID" begin
        topo = Tet4()
        
        # Position in vector IS the entity ID
        edges_list = edges(topo)
        @test edges_list[1].vertices == (1, 2)  # Edge ID 1
        @test edges_list[2].vertices == (2, 3)  # Edge ID 2
        @test edges_list[6].vertices == (3, 4)  # Edge ID 6
        
        # No need for explicit id field
        @test !hasproperty(edges_list[1], :id)
    end
end
