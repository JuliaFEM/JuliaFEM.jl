# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using Test
using Tensors

# Mock topology types
abstract type AbstractTopology end
struct Tet4 <: AbstractTopology end
nnodes(::Type{Tet4}) = 4

# Include mesh implementation
include("../src/mesh/mesh.jl")

@testset "Mesh Parallel Features - Naming, Coloring, Permutation, Ghosts" begin

    # Setup: Create a simple two-element mesh
    function make_test_mesh()
        nodes = [
            Vec(0.0, 0.0, 0.0),  # 1
            Vec(1.0, 0.0, 0.0),  # 2
            Vec(0.0, 1.0, 0.0),  # 3
            Vec(0.0, 0.0, 1.0),  # 4
            Vec(1.0, 1.0, 0.0)   # 5
        ]
        connectivity = [
            (UInt32(1), UInt32(2), UInt32(3), UInt32(4)),  # Element 1
            (UInt32(2), UInt32(5), UInt32(3), UInt32(4))   # Element 2
        ]
        return Mesh{Tet4}(nodes, connectivity)
    end

    @testset "Node Naming - Industrial ID Ranges" begin
        mesh = make_test_mesh()

        # Assign industrial ID ranges (part 1: 10M+, part 2: 20M+)
        set_node_id!(mesh, UInt32(1), 10_000_001)
        set_node_id!(mesh, UInt32(2), 10_000_002)
        set_node_id!(mesh, UInt32(3), 20_000_001)

        # Retrieve by ID
        @test get_node_by_id(mesh, 10_000_001) == UInt32(1)
        @test get_node_by_id(mesh, 10_000_002) == UInt32(2)
        @test get_node_by_id(mesh, 20_000_001) == UInt32(3)

        # Non-existent ID should error
        @test_throws AssertionError get_node_by_id(mesh, 99999)
    end

    @testset "Node Naming - Symbolic Names (Code Aster Style)" begin
        mesh = make_test_mesh()

        # Assign symbolic names
        set_node_id!(mesh, UInt32(1), :N1)
        set_node_id!(mesh, UInt32(2), :N2)
        set_node_id!(mesh, UInt32(5), :corner_node)

        # Retrieve by symbol
        @test get_node_by_id(mesh, :N1) == UInt32(1)
        @test get_node_by_id(mesh, :N2) == UInt32(2)
        @test get_node_by_id(mesh, :corner_node) == UInt32(5)

        # Non-existent symbol should error
        @test_throws AssertionError get_node_by_id(mesh, :nonexistent)
    end

    @testset "Element Naming - Industrial ID Ranges" begin
        mesh = make_test_mesh()

        # Assign element IDs
        set_element_id!(mesh, UInt32(1), 30_000_001)
        set_element_id!(mesh, UInt32(2), 30_000_002)

        # Retrieve by ID
        @test get_element_by_id(mesh, 30_000_001) == UInt32(1)
        @test get_element_by_id(mesh, 30_000_002) == UInt32(2)

        # Non-existent ID should error
        @test_throws AssertionError get_element_by_id(mesh, 99999)
    end

    @testset "Element Naming - Symbolic Names" begin
        mesh = make_test_mesh()

        # Assign symbolic names
        set_element_id!(mesh, UInt32(1), :E1)
        set_element_id!(mesh, UInt32(2), :E2)

        # Retrieve by symbol
        @test get_element_by_id(mesh, :E1) == UInt32(1)
        @test get_element_by_id(mesh, :E2) == UInt32(2)
    end

    @testset "Node Coloring - Load Balancing" begin
        mesh = make_test_mesh()

        # Initially all nodes uncolored (color = 0)
        for i in 1:nnodes_total(mesh)
            @test get_node_color(mesh, UInt32(i)) == UInt32(0)
        end

        # Assign nodes to MPI ranks (colors 1-4)
        for i in 1:nnodes_total(mesh)
            rank = mod(i - 1, 4) + 1  # Round-robin: 1,2,3,4,1
            set_node_color!(mesh, UInt32(i), UInt32(rank))
        end

        # Verify colors
        @test get_node_color(mesh, UInt32(1)) == UInt32(1)
        @test get_node_color(mesh, UInt32(2)) == UInt32(2)
        @test get_node_color(mesh, UInt32(3)) == UInt32(3)
        @test get_node_color(mesh, UInt32(4)) == UInt32(4)
        @test get_node_color(mesh, UInt32(5)) == UInt32(1)
    end

    @testset "Element Coloring - Graph Coloring for Threading" begin
        mesh = make_test_mesh()

        # Initially all elements uncolored
        @test get_element_color(mesh, UInt32(1)) == UInt32(0)
        @test get_element_color(mesh, UInt32(2)) == UInt32(0)

        # Assign colors (elements sharing nodes get different colors)
        set_element_color!(mesh, UInt32(1), UInt32(1))
        set_element_color!(mesh, UInt32(2), UInt32(2))  # Shares nodes with elem 1

        # Verify colors
        @test get_element_color(mesh, UInt32(1)) == UInt32(1)
        @test get_element_color(mesh, UInt32(2)) == UInt32(2)

        # Get elements by color
        color1_elems = get_elements_with_color(mesh, UInt32(1))
        color2_elems = get_elements_with_color(mesh, UInt32(2))

        @test UInt32(1) in color1_elems
        @test UInt32(2) in color2_elems
        @test length(color1_elems) == 1
        @test length(color2_elems) == 1
    end

    @testset "Ghost Nodes - MPI Domain Decomposition" begin
        mesh = make_test_mesh()

        # Initially no ghost nodes
        @test !is_ghost_node(mesh, UInt32(1))
        @test !is_ghost_node(mesh, UInt32(2))

        # Mark nodes 2,3,4 as ghosts (owned by another rank)
        mark_ghost_node!(mesh, UInt32(2))
        mark_ghost_node!(mesh, UInt32(3))
        mark_ghost_node!(mesh, UInt32(4))

        # Verify ghost status
        @test !is_ghost_node(mesh, UInt32(1))  # Local
        @test is_ghost_node(mesh, UInt32(2))   # Ghost
        @test is_ghost_node(mesh, UInt32(3))   # Ghost
        @test is_ghost_node(mesh, UInt32(4))   # Ghost
        @test !is_ghost_node(mesh, UInt32(5))  # Local

        # Get local nodes (non-ghost)
        local_nodes = get_local_nodes(mesh)
        @test UInt32(1) in local_nodes
        @test UInt32(5) in local_nodes
        @test !(UInt32(2) in local_nodes)
        @test !(UInt32(3) in local_nodes)
        @test !(UInt32(4) in local_nodes)
        @test length(local_nodes) == 2
    end

    @testset "Ghost Elements - MPI Domain Decomposition" begin
        mesh = make_test_mesh()

        # Initially no ghost elements
        @test !is_ghost_element(mesh, UInt32(1))
        @test !is_ghost_element(mesh, UInt32(2))

        # Mark element 2 as ghost
        mark_ghost_element!(mesh, UInt32(2))

        # Verify ghost status
        @test !is_ghost_element(mesh, UInt32(1))  # Local
        @test is_ghost_element(mesh, UInt32(2))   # Ghost

        # Get local elements
        local_elems = get_local_elements(mesh)
        @test UInt32(1) in local_elems
        @test !(UInt32(2) in local_elems)
        @test length(local_elems) == 1
    end

    @testset "Node Permutation - Identity (Initial State)" begin
        mesh = make_test_mesh()

        # Initially identity permutation
        for i in 1:nnodes_total(mesh)
            @test mesh.node_permutation[i] == UInt32(i)
            @test mesh.node_inverse_permutation[i] == UInt32(i)
        end

        # Forward and inverse should be consistent
        for i in 1:nnodes_total(mesh)
            j = get_reordered_node_index(mesh, UInt32(i))
            @test get_original_node_index(mesh, j) == UInt32(i)
        end
    end

    @testset "Node Permutation - Custom Reordering" begin
        mesh = make_test_mesh()

        # Apply custom permutation (reverse order for simplicity)
        n = nnodes_total(mesh)
        perm = UInt32[n, n-1, n-2, n-3, n-4]  # [5, 4, 3, 2, 1]
        apply_node_permutation!(mesh, perm)

        # Verify permutation
        @test mesh.node_permutation == perm

        # Verify inverse permutation
        @test mesh.node_inverse_permutation == UInt32[5, 4, 3, 2, 1]

        # Check forward mapping: original 1 → reordered 5
        @test get_reordered_node_index(mesh, UInt32(1)) == UInt32(5)
        @test get_reordered_node_index(mesh, UInt32(5)) == UInt32(1)

        # Check inverse mapping: reordered 1 → original 5
        @test get_original_node_index(mesh, UInt32(1)) == UInt32(5)
        @test get_original_node_index(mesh, UInt32(5)) == UInt32(1)

        # Verify consistency
        for i in 1:n
            j = get_reordered_node_index(mesh, UInt32(i))
            @test get_original_node_index(mesh, j) == UInt32(i)
        end
    end

    @testset "Node Permutation - Invalid Permutation" begin
        mesh = make_test_mesh()

        # Wrong size
        @test_throws AssertionError apply_node_permutation!(mesh, UInt32[1, 2])

        # Invalid permutation (duplicate)
        @test_throws AssertionError apply_node_permutation!(mesh, UInt32[1, 1, 2, 3, 4])

        # Invalid permutation (out of range)
        @test_throws AssertionError apply_node_permutation!(mesh, UInt32[1, 2, 3, 4, 6])
    end

    @testset "Element Permutation - Cache Optimization" begin
        mesh = make_test_mesh()

        # Initially identity
        @test mesh.element_permutation == UInt32[1, 2]

        # Apply custom permutation (swap elements)
        perm = UInt32[2, 1]
        apply_element_permutation!(mesh, perm)

        # Verify permutation
        @test mesh.element_permutation == perm
    end

    @testset "Combined Features - Industrial Workflow" begin
        mesh = make_test_mesh()

        # 1. Assign industrial IDs (multi-part assembly)
        set_node_id!(mesh, UInt32(1), 10_000_001)
        set_node_id!(mesh, UInt32(2), 10_000_002)
        set_node_id!(mesh, UInt32(3), 20_000_001)  # Part 2 starts here

        set_element_id!(mesh, UInt32(1), 30_000_001)
        set_element_id!(mesh, UInt32(2), 30_000_002)

        # 2. Apply bandwidth minimization (RCM-like)
        perm = UInt32[2, 1, 3, 4, 5]  # Simulated RCM result
        apply_node_permutation!(mesh, perm)

        # 3. Color elements for parallel assembly
        set_element_color!(mesh, UInt32(1), UInt32(1))
        set_element_color!(mesh, UInt32(2), UInt32(2))

        # 4. Mark ghost nodes (MPI partitioning)
        mark_ghost_node!(mesh, UInt32(3))
        mark_ghost_node!(mesh, UInt32(4))

        # Verify everything works together
        @test get_node_by_id(mesh, 10_000_001) == UInt32(1)
        @test get_element_by_id(mesh, 30_000_001) == UInt32(1)
        @test get_reordered_node_index(mesh, UInt32(1)) == UInt32(2)
        @test get_element_color(mesh, UInt32(1)) == UInt32(1)
        @test is_ghost_node(mesh, UInt32(3))
        @test length(get_local_nodes(mesh)) == 3  # 5 nodes - 2 ghosts = 3 local
    end

end

println("✅ All parallel features tests passed!")
