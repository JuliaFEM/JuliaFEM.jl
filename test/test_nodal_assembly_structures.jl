# Test nodal assembly data structures
#
# Demonstrates the "spider" pattern and inverse mapping for nodal assembly

using Test
using Tensors

include("../src/nodal_assembly_structures.jl")

@testset "Nodal Assembly Structures" begin

    @testset "Simple Tet4 Mesh - 2 Elements" begin
        # Two tetrahedra sharing a face (nodes 2,3,4)
        #
        #     5
        #    /|\
        #   / | \
        #  /  |  \
        # 1---2---4
        #  \  |  /
        #   \ | /
        #    \|/
        #     3
        #
        # Element 1: nodes (1,2,3,4)
        # Element 2: nodes (2,3,4,5)

        connectivity = [
            (1, 2, 3, 4),  # Element 1
            (2, 3, 4, 5)   # Element 2
        ]

        # Build inverse mapping
        map = NodeToElementsMap(connectivity)

        @test map.nnodes == 5
        @test map.nelements == 2

        # Node 1: only in element 1
        @test length(map.node_to_elements[1]) == 1
        @test map.node_to_elements[1][1].element_id == 1
        @test map.node_to_elements[1][1].local_node_idx == 1

        # Node 2: in both elements
        @test length(map.node_to_elements[2]) == 2
        @test map.node_to_elements[2][1].element_id == 1
        @test map.node_to_elements[2][1].local_node_idx == 2
        @test map.node_to_elements[2][2].element_id == 2
        @test map.node_to_elements[2][2].local_node_idx == 1

        # Node 5: only in element 2
        @test length(map.node_to_elements[5]) == 1
        @test map.node_to_elements[5][1].element_id == 2
        @test map.node_to_elements[5][1].local_node_idx == 4
    end

    @testset "Node Spider - Coupling Pattern" begin
        connectivity = [
            (1, 2, 3, 4),
            (2, 3, 4, 5)
        ]

        map = NodeToElementsMap(connectivity)

        # Node 1 spider: only element 1 touches it
        spider_1 = get_node_spider(map, 1, connectivity)
        @test spider_1 == [1, 2, 3, 4]
        @test length(spider_1) == 4  # 4 nodes in spider → 4×3x3 blocks

        # Node 2 spider: both elements touch it
        spider_2 = get_node_spider(map, 2, connectivity)
        @test spider_2 == [1, 2, 3, 4, 5]  # Union of both elements
        @test length(spider_2) == 5  # 5 nodes in spider → 5×3x3 blocks

        # Node 3 spider: both elements
        spider_3 = get_node_spider(map, 3, connectivity)
        @test spider_3 == [1, 2, 3, 4, 5]

        # Node 5 spider: only element 2
        spider_5 = get_node_spider(map, 5, connectivity)
        @test spider_5 == [2, 3, 4, 5]
        @test length(spider_5) == 4
    end

    @testset "NodalStiffnessContribution - Storage" begin
        connectivity = [(1, 2, 3, 4)]
        map = NodeToElementsMap(connectivity)
        spider = get_node_spider(map, 1, connectivity)

        # Allocate contribution storage
        contrib = NodalStiffnessContribution(1, spider, Float64)

        @test contrib.node_id == 1
        @test contrib.spider_nodes == [1, 2, 3, 4]
        @test length(contrib.K_blocks) == 4
        @test all(iszero, contrib.K_blocks)  # Initially zero
        @test iszero(contrib.f_int)
        @test iszero(contrib.f_ext)

        # Verify types (Tensors.jl)
        @test contrib.K_blocks[1] isa Tensor{2,3,Float64,9}
        @test contrib.f_int isa Vec{3,Float64}
        @test contrib.f_ext isa Vec{3,Float64}
    end

    @testset "Matrix-Vector Product - Nodal" begin
        # Simple 2-node problem for testing
        connectivity = [(1, 2, 3, 4)]
        map = NodeToElementsMap(connectivity)
        spider = get_node_spider(map, 1, connectivity)

        contrib = NodalStiffnessContribution(1, spider, Float64)

        # Fill in some test stiffness blocks
        # K_11 = identity (self-coupling)
        contrib.K_blocks[1] = one(Tensor{2,3,Float64})

        # K_12 = simple coupling
        contrib.K_blocks[2] = Tensor{2,3}((2.0, 0.0, 0.0,
            0.0, 2.0, 0.0,
            0.0, 0.0, 2.0))

        # K_13, K_14 = zero (no coupling for this test)
        contrib.K_blocks[3] = zero(Tensor{2,3,Float64})
        contrib.K_blocks[4] = zero(Tensor{2,3,Float64})

        # Displacement field
        u = [
            Vec{3}((1.0, 2.0, 3.0)),  # u_1
            Vec{3}((0.5, 0.5, 0.5)),  # u_2
            Vec{3}((0.0, 0.0, 0.0)),  # u_3
            Vec{3}((0.0, 0.0, 0.0))   # u_4
        ]

        # Matrix-vector product at node 1
        w_1 = matrix_vector_product_nodal(contrib, u)

        # Expected: w_1 = K_11*u_1 + K_12*u_2
        # = [1 0 0; 0 1 0; 0 0 1] * [1,2,3] + [2 0 0; 0 2 0; 0 0 2] * [0.5,0.5,0.5]
        # = [1,2,3] + [1,1,1] = [2,3,4]

        expected = Vec{3}((2.0, 3.0, 4.0))
        @test w_1 ≈ expected
    end

    @testset "Spider Efficiency Check" begin
        # Larger mesh to show efficiency
        # Hex8 mesh: 2×2×2 elements = 27 nodes
        # Build connectivity for structured hex mesh
        connectivity = Vector{NTuple{8,Int}}()

        node_id(i, j, k) = 1 + i + 3 * j + 9 * k  # 3×3×3 grid

        for iz in 0:1, iy in 0:1, ix in 0:1
            push!(connectivity, (
                node_id(ix, iy, iz),
                node_id(ix + 1, iy, iz),
                node_id(ix + 1, iy + 1, iz),
                node_id(ix, iy + 1, iz),
                node_id(ix, iy, iz + 1),
                node_id(ix + 1, iy, iz + 1),
                node_id(ix + 1, iy + 1, iz + 1),
                node_id(ix, iy + 1, iz + 1)
            ))
        end

        map = NodeToElementsMap(connectivity)
        @test map.nelements == 8

        # Corner node: touches 1 element
        corner_node = node_id(0, 0, 0)
        spider_corner = get_node_spider(map, corner_node, connectivity)
        @test length(spider_corner) == 8  # Only nodes in 1 hex

        # Center node: touches 8 elements
        center_node = node_id(1, 1, 1)
        spider_center = get_node_spider(map, center_node, connectivity)
        @test length(spider_center) == 27  # All nodes in mesh!

        # Edge node: touches fewer elements
        edge_node = node_id(1, 0, 0)
        spider_edge = get_node_spider(map, edge_node, connectivity)
        @test 8 < length(spider_edge) < 27  # Intermediate

        println("\nSpider sizes for 2×2×2 hex mesh (27 nodes):")
        println("  Corner node: $(length(spider_corner)) couplings")
        println("  Edge node:   $(length(spider_edge)) couplings")
        println("  Center node: $(length(spider_center)) couplings")
        println("\nKey insight: We only compute non-zero blocks!")
        println("  Full matrix would be 81×81 (27 nodes × 3 DOF)")
        println("  But most nodes have < 27 couplings")
    end

    @testset "Diagnostic Output" begin
        connectivity = [
            (1, 2, 3, 4),
            (2, 3, 4, 5)
        ]

        map = NodeToElementsMap(connectivity)

        println("\n" * "="^60)
        println("DIAGNOSTIC: Spider pattern for node 2 (shared by 2 elements)")
        println("="^60)
        print_spider_info(map, 2, connectivity)
        println("="^60)
    end
end
