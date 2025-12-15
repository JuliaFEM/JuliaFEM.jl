# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using Test
using JuliaFEM
using Tensors

@testset "Mesh refinement" begin
    @testset "Single Hex8 refinement - 1 level" begin
        # Create a single hex element: unit cube [0,1]³
        nodes = [
            Vec(0.0, 0.0, 0.0), Vec(1.0, 0.0, 0.0),
            Vec(1.0, 1.0, 0.0), Vec(0.0, 1.0, 0.0),
            Vec(0.0, 0.0, 1.0), Vec(1.0, 0.0, 1.0),
            Vec(1.0, 1.0, 1.0), Vec(0.0, 1.0, 1.0)
        ]
        connectivity = [(UInt32(1), UInt32(2), UInt32(3), UInt32(4),
            UInt32(5), UInt32(6), UInt32(7), UInt32(8))]

        mesh = Mesh{Hex8}(nodes, connectivity)

        # Refine once
        strategy = LongestEdgeBisection(1)
        refined = refine(mesh, strategy)

        # After 1 refinement: 1 element → 2 elements
        @test nelements(refined) == 2

        # Should have original 8 nodes + 4 new midpoint nodes = 12 nodes
        @test nnodes_total(refined) == 12

        # Check that all elements have 8 nodes
        for conn in refined.connectivity
            @test length(conn) == 8
        end

        println("1 level refinement: ", nelements(mesh), " → ", nelements(refined), " elements")
    end

    @testset "Single Hex8 refinement - 2 levels" begin
        # Create a single hex element
        nodes = [
            Vec(0.0, 0.0, 0.0), Vec(1.0, 0.0, 0.0),
            Vec(1.0, 1.0, 0.0), Vec(0.0, 1.0, 0.0),
            Vec(0.0, 0.0, 1.0), Vec(1.0, 0.0, 1.0),
            Vec(1.0, 1.0, 1.0), Vec(0.0, 1.0, 1.0)
        ]
        connectivity = [(UInt32(1), UInt32(2), UInt32(3), UInt32(4),
            UInt32(5), UInt32(6), UInt32(7), UInt32(8))]

        mesh = Mesh{Hex8}(nodes, connectivity)

        # Refine twice
        strategy = LongestEdgeBisection(2)
        refined = refine(mesh, strategy)

        # After 2 refinements: 1 → 2 → 4 elements
        @test nelements(refined) == 4

        println("2 level refinement: ", nelements(mesh), " → ", nelements(refined), " elements")
    end

    @testset "Single Hex8 refinement - 3 levels" begin
        # Create a single hex element
        nodes = [
            Vec(0.0, 0.0, 0.0), Vec(1.0, 0.0, 0.0),
            Vec(1.0, 1.0, 0.0), Vec(0.0, 1.0, 0.0),
            Vec(0.0, 0.0, 1.0), Vec(1.0, 0.0, 1.0),
            Vec(1.0, 1.0, 1.0), Vec(0.0, 1.0, 1.0)
        ]
        connectivity = [(UInt32(1), UInt32(2), UInt32(3), UInt32(4),
            UInt32(5), UInt32(6), UInt32(7), UInt32(8))]

        mesh = Mesh{Hex8}(nodes, connectivity)

        # Refine 3 times
        strategy = LongestEdgeBisection(3)
        refined = refine(mesh, strategy)

        # After 3 refinements: 1 → 2 → 4 → 8 elements
        @test nelements(refined) == 8

        println("3 level refinement: ", nelements(mesh), " → ", nelements(refined), " elements")
    end

    @testset "Rectangular Hex8 refinement" begin
        # Create elongated hex element: [0,2] × [0,1] × [0,1]
        # Should split along X first (longest dimension)
        nodes = [
            Vec(0.0, 0.0, 0.0), Vec(2.0, 0.0, 0.0),
            Vec(2.0, 1.0, 0.0), Vec(0.0, 1.0, 0.0),
            Vec(0.0, 0.0, 1.0), Vec(2.0, 0.0, 1.0),
            Vec(2.0, 1.0, 1.0), Vec(0.0, 1.0, 1.0)
        ]
        connectivity = [(UInt32(1), UInt32(2), UInt32(3), UInt32(4),
            UInt32(5), UInt32(6), UInt32(7), UInt32(8))]

        mesh = Mesh{Hex8}(nodes, connectivity)
        strategy = LongestEdgeBisection(1)
        refined = refine(mesh, strategy)

        @test nelements(refined) == 2

        # Check that split happened along X (should create nodes near x=1.0)
        # New nodes should be at midpoints
        midpoint_nodes = refined.nodes[9:12]  # Last 4 nodes are new
        for node in midpoint_nodes
            # All midpoint nodes should have x ≈ 1.0 (midpoint of [0,2])
            @test node[1] ≈ 1.0
        end

        println("Rectangular refinement along longest dimension successful")
    end

    @testset "Multiple element refinement" begin
        # Create 2 hex elements side by side
        nodes = [
            Vec(0.0, 0.0, 0.0), Vec(1.0, 0.0, 0.0),
            Vec(1.0, 1.0, 0.0), Vec(0.0, 1.0, 0.0),
            Vec(0.0, 0.0, 1.0), Vec(1.0, 0.0, 1.0),
            Vec(1.0, 1.0, 1.0), Vec(0.0, 1.0, 1.0),
            # Second element nodes
            Vec(2.0, 0.0, 0.0), Vec(2.0, 1.0, 0.0),
            Vec(2.0, 0.0, 1.0), Vec(2.0, 1.0, 1.0)
        ]
        connectivity = [
            (UInt32(1), UInt32(2), UInt32(3), UInt32(4),
                UInt32(5), UInt32(6), UInt32(7), UInt32(8)),
            (UInt32(2), UInt32(9), UInt32(10), UInt32(3),
                UInt32(6), UInt32(11), UInt32(12), UInt32(7))
        ]

        mesh = Mesh{Hex8}(nodes, connectivity)
        strategy = LongestEdgeBisection(1)
        refined = refine(mesh, strategy)

        # 2 elements → 4 elements after 1 refinement
        @test nelements(refined) == 4

        println("Multiple element refinement: ", nelements(mesh), " → ", nelements(refined), " elements")
    end

    @testset "Element set preservation" begin
        # Create mesh with element sets
        nodes = [
            Vec(0.0, 0.0, 0.0), Vec(1.0, 0.0, 0.0),
            Vec(1.0, 1.0, 0.0), Vec(0.0, 1.0, 0.0),
            Vec(0.0, 0.0, 1.0), Vec(1.0, 0.0, 1.0),
            Vec(1.0, 1.0, 1.0), Vec(0.0, 1.0, 1.0)
        ]
        connectivity = [(UInt32(1), UInt32(2), UInt32(3), UInt32(4),
            UInt32(5), UInt32(6), UInt32(7), UInt32(8))]

        element_sets = Dict(:volume => Set(UInt32[1]))
        mesh = Mesh{Hex8}(nodes, connectivity, element_sets)

        strategy = LongestEdgeBisection(2)
        refined = refine(mesh, strategy)

        # Element set should be preserved and expanded
        @test haskey(refined.element_sets, :volume)
        @test length(refined.element_sets[:volume]) == 4  # 1 → 2 → 4 elements

        println("Element set preservation: OK")
    end
end
