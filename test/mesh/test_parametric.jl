# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using Test
using Tensors

# We need topology definitions - for now use mock types
abstract type AbstractTopology end
struct Tet4 <: AbstractTopology end
struct Tet10 <: AbstractTopology end
struct Hex8 <: AbstractTopology end
struct Tri3 <: AbstractTopology end
struct Tri6 <: AbstractTopology end
struct Quad4 <: AbstractTopology end
struct Seg2 <: AbstractTopology end

# Topology interface
nnodes(::Type{Tet4}) = 4
nnodes(::Type{Tet10}) = 10
nnodes(::Type{Hex8}) = 8
nnodes(::Type{Tri3}) = 3
nnodes(::Type{Tri6}) = 6
nnodes(::Type{Quad4}) = 4
nnodes(::Type{Seg2}) = 2

surface_topology(::Type{Tet4}) = Tri3
surface_topology(::Type{Tet10}) = Tri6
surface_topology(::Type{Hex8}) = Quad4

# Include mesh implementation
include("../src/mesh/mesh.jl")

@testset "Mesh{T} Parametric - Production Implementation" begin

    @testset "Construction - Basic" begin
        # Tet4 mesh (single element)
        nodes = [Vec(0.0, 0.0, 0.0), Vec(1.0, 0.0, 0.0),
            Vec(0.0, 1.0, 0.0), Vec(0.0, 0.0, 1.0)]
        connectivity = [(UInt32(1), UInt32(2), UInt32(3), UInt32(4))]
        mesh = Mesh{Tet4}(nodes, connectivity)

        @test mesh isa Mesh{Tet4}
        @test nnodes_total(mesh) == 4
        @test nelements(mesh) == 1
        @test topology_type(mesh) == Tet4
        @test nnodes_per_element(mesh) == 4
    end

    @testset "Construction - Multiple Elements" begin
        # Two Tet4 elements sharing nodes
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
        mesh = Mesh{Tet4}(nodes, connectivity)

        @test nnodes_total(mesh) == 5
        @test nelements(mesh) == 2
        @test length(mesh.connectivity[1]) == 4
        @test length(mesh.connectivity[2]) == 4
    end

    @testset "Construction - With Sets" begin
        nodes = [Vec(0.0, 0.0, 0.0), Vec(1.0, 0.0, 0.0),
            Vec(0.0, 1.0, 0.0), Vec(0.0, 0.0, 1.0)]
        connectivity = [(UInt32(1), UInt32(2), UInt32(3), UInt32(4))]

        element_sets = Dict(:all => Set(UInt32[1]), :body => Set(UInt32[1]))
        node_sets = Dict(:corner => Set(UInt32[1]), :boundary => Set(UInt32[1, 2, 3]))

        mesh = Mesh{Tet4}(nodes, connectivity, element_sets, node_sets)

        @test haskey(mesh.element_sets, :all)
        @test haskey(mesh.element_sets, :body)
        @test haskey(mesh.node_sets, :corner)
        @test haskey(mesh.node_sets, :boundary)
        @test length(mesh.element_sets[:all]) == 1
        @test length(mesh.node_sets[:boundary]) == 3
    end

    @testset "Construction - Keyword Arguments" begin
        nodes = [Vec(0.0, 0.0, 0.0), Vec(1.0, 0.0, 0.0),
            Vec(0.0, 1.0, 0.0), Vec(0.0, 0.0, 1.0)]
        connectivity = [(UInt32(1), UInt32(2), UInt32(3), UInt32(4))]

        mesh = Mesh{Tet4}(nodes, connectivity;
            element_sets=Dict(:all => Set(UInt32[1])),
            node_sets=Dict(:corner => Set(UInt32[1])))

        @test haskey(mesh.element_sets, :all)
        @test haskey(mesh.node_sets, :corner)
    end

    @testset "Validation - Connectivity Size Mismatch" begin
        nodes = [Vec(0.0, 0.0, 0.0), Vec(1.0, 0.0, 0.0),
            Vec(0.0, 1.0, 0.0)]
        connectivity = [(UInt32(1), UInt32(2), UInt32(3))]  # Only 3 nodes, but Tet4 needs 4!

        @test_throws AssertionError Mesh{Tet4}(nodes, connectivity)
    end

    @testset "Validation - Node Index Out of Range" begin
        nodes = [Vec(0.0, 0.0, 0.0), Vec(1.0, 0.0, 0.0),
            Vec(0.0, 1.0, 0.0), Vec(0.0, 0.0, 1.0)]
        connectivity = [(UInt32(1), UInt32(2), UInt32(3), UInt32(5))]  # Node 5 doesn't exist!

        @test_throws AssertionError Mesh{Tet4}(nodes, connectivity)
    end

    @testset "Validation - Element Set Out of Range" begin
        nodes = [Vec(0.0, 0.0, 0.0), Vec(1.0, 0.0, 0.0),
            Vec(0.0, 1.0, 0.0), Vec(0.0, 0.0, 1.0)]
        connectivity = [(UInt32(1), UInt32(2), UInt32(3), UInt32(4))]
        element_sets = Dict(:all => Set(UInt32[1, 2]))  # Element 2 doesn't exist!

        @test_throws AssertionError Mesh{Tet4}(nodes, connectivity, element_sets)
    end

    @testset "Connectivity Matrix - Tet4" begin
        nodes = [
            Vec(0.0, 0.0, 0.0),
            Vec(1.0, 0.0, 0.0),
            Vec(0.0, 1.0, 0.0),
            Vec(0.0, 0.0, 1.0),
            Vec(1.0, 1.0, 0.0)
        ]
        connectivity = [
            (UInt32(1), UInt32(2), UInt32(3), UInt32(4)),
            (UInt32(2), UInt32(5), UInt32(3), UInt32(4))
        ]
        mesh = Mesh{Tet4}(nodes, connectivity)

        conn_mat = connectivity_matrix(mesh)

        @test size(conn_mat) == (4, 2)  # 4 nodes/element, 2 elements
        @test conn_mat[:, 1] == UInt32[1, 2, 3, 4]
        @test conn_mat[:, 2] == UInt32[2, 5, 3, 4]
    end

    @testset "Connectivity Matrix - Tet10" begin
        # Create minimal Tet10 mesh
        nodes = [Vec(Float64(i - 1), 0.0, 0.0) for i in 1:10]
        connectivity = [ntuple(i -> UInt32(i), 10)]
        mesh = Mesh{Tet10}(nodes, connectivity)

        conn_mat = connectivity_matrix(mesh)

        @test size(conn_mat) == (10, 1)  # 10 nodes/element, 1 element
        @test conn_mat[:, 1] == UInt32.(1:10)
    end

    @testset "Inverse Connectivity - Single Element" begin
        # Tet4 with single element
        nodes = [
            Vec(0.0, 0.0, 0.0),  # 1
            Vec(1.0, 0.0, 0.0),  # 2
            Vec(0.0, 1.0, 0.0),  # 3
            Vec(0.0, 0.0, 1.0)   # 4
        ]
        connectivity = [(UInt32(1), UInt32(2), UInt32(3), UInt32(4))]
        mesh = Mesh{Tet4}(nodes, connectivity)

        # Each node should appear in exactly 1 element (element 1)
        for node_id in 1:4
            elems = get_elements_for_node(mesh, node_id)
            @test length(elems) == 1
            @test elems[1][1] == UInt32(1)  # Element ID
            @test elems[1][2] == UInt8(node_id)  # Local index matches node_id for this simple case
        end
    end

    @testset "Inverse Connectivity - Shared Nodes" begin
        # Two Tet4 elements sharing nodes
        nodes = [
            Vec(0.0, 0.0, 0.0),  # 1 - in both elements
            Vec(1.0, 0.0, 0.0),  # 2 - in both elements
            Vec(0.0, 1.0, 0.0),  # 3 - in both elements
            Vec(0.0, 0.0, 1.0),  # 4 - in both elements
            Vec(1.0, 1.0, 0.0)   # 5 - only in element 2
        ]
        connectivity = [
            (UInt32(1), UInt32(2), UInt32(3), UInt32(4)),  # Element 1
            (UInt32(2), UInt32(5), UInt32(3), UInt32(4))   # Element 2
        ]
        mesh = Mesh{Tet4}(nodes, connectivity)

        # Node 1: only in element 1
        elems_1 = get_elements_for_node(mesh, 1)
        @test length(elems_1) == 1
        @test elems_1[1][1] == UInt32(1)  # Element 1
        @test elems_1[1][2] == UInt8(1)   # Local index 1

        # Node 2: in both elements (index 2 in elem 1, index 1 in elem 2)
        elems_2 = get_elements_for_node(mesh, 2)
        @test length(elems_2) == 2
        @test (UInt32(1), UInt8(2)) in elems_2  # Element 1, local index 2
        @test (UInt32(2), UInt8(1)) in elems_2  # Element 2, local index 1

        # Node 3: in both elements (index 3 in both)
        elems_3 = get_elements_for_node(mesh, 3)
        @test length(elems_3) == 2
        @test (UInt32(1), UInt8(3)) in elems_3
        @test (UInt32(2), UInt8(3)) in elems_3

        # Node 4: in both elements (index 4 in both)
        elems_4 = get_elements_for_node(mesh, 4)
        @test length(elems_4) == 2
        @test (UInt32(1), UInt8(4)) in elems_4
        @test (UInt32(2), UInt8(4)) in elems_4

        # Node 5: only in element 2 (index 2)
        elems_5 = get_elements_for_node(mesh, 5)
        @test length(elems_5) == 1
        @test elems_5[1][1] == UInt32(2)  # Element 2
        @test elems_5[1][2] == UInt8(2)   # Local index 2
    end

    @testset "Inverse Connectivity - Nodal Assembly Pattern" begin
        # Verify inverse connectivity enables nodal assembly
        nodes = [
            Vec(0.0, 0.0, 0.0),
            Vec(1.0, 0.0, 0.0),
            Vec(0.0, 1.0, 0.0),
            Vec(0.0, 0.0, 1.0),
            Vec(1.0, 1.0, 0.0)
        ]
        connectivity = [
            (UInt32(1), UInt32(2), UInt32(3), UInt32(4)),
            (UInt32(2), UInt32(5), UInt32(3), UInt32(4))
        ]
        mesh = Mesh{Tet4}(nodes, connectivity)

        # Simulate nodal assembly: for each node, iterate over connected elements
        node_element_counts = zeros(Int, 5)
        for node_i in 1:nnodes_total(mesh)
            for (elem_id, local_idx) in get_elements_for_node(mesh, node_i)
                # Verify we can access element connectivity
                elem_conn = mesh.connectivity[elem_id]
                # Verify local index is correct
                @test elem_conn[local_idx] == UInt32(node_i)
                node_element_counts[node_i] += 1
            end
        end

        # Verify counts
        @test node_element_counts == [1, 2, 2, 2, 1]  # Nodes 2,3,4 shared by both elements
    end

    @testset "Node Operations - Get Node" begin
        nodes = [Vec(0.0, 0.0, 0.0), Vec(1.0, 2.0, 3.0),
            Vec(4.0, 5.0, 6.0), Vec(7.0, 8.0, 9.0)]
        connectivity = [(UInt32(1), UInt32(2), UInt32(3), UInt32(4))]
        mesh = Mesh{Tet4}(nodes, connectivity)

        @test get_node(mesh, 1) == Vec(0.0, 0.0, 0.0)
        @test get_node(mesh, 2) == Vec(1.0, 2.0, 3.0)
        @test get_node(mesh, 4) == Vec(7.0, 8.0, 9.0)

        @test_throws AssertionError get_node(mesh, 0)
        @test_throws AssertionError get_node(mesh, 5)
    end

    @testset "Node Operations - Find Nearest Node" begin
        nodes = [
            Vec(0.0, 0.0, 0.0),  # 1
            Vec(1.0, 0.0, 0.0),  # 2
            Vec(0.0, 1.0, 0.0),  # 3
            Vec(0.0, 0.0, 1.0)   # 4
        ]
        connectivity = [(UInt32(1), UInt32(2), UInt32(3), UInt32(4))]
        mesh = Mesh{Tet4}(nodes, connectivity)

        # Find nearest to (0.1, 0.0, 0.0) - should be node 1
        nearest = find_nearest_node(mesh, Vec(0.1, 0.0, 0.0))
        @test nearest == UInt32(1)

        # Find nearest to (0.9, 0.0, 0.0) - should be node 2
        nearest = find_nearest_node(mesh, Vec(0.9, 0.0, 0.0))
        @test nearest == UInt32(2)

        # Find nearest to (0.0, 0.8, 0.0) - should be node 3
        nearest = find_nearest_node(mesh, Vec(0.0, 0.8, 0.0))
        @test nearest == UInt32(3)
    end

    @testset "Node Operations - Find Nearest Nodes (Multiple)" begin
        nodes = [
            Vec(0.0, 0.0, 0.0),  # 1
            Vec(1.0, 0.0, 0.0),  # 2
            Vec(0.0, 1.0, 0.0),  # 3
            Vec(0.0, 0.0, 1.0)   # 4
        ]
        connectivity = [(UInt32(1), UInt32(2), UInt32(3), UInt32(4))]
        mesh = Mesh{Tet4}(nodes, connectivity)

        # Find 2 nearest to origin
        nearest = find_nearest_nodes(mesh, Vec(0.0, 0.0, 0.0), 2)
        @test length(nearest) == 2
        @test nearest[1] == UInt32(1)  # Closest
        @test nearest[2] in UInt32[2, 3, 4]  # All equidistant

        # Find all 4 nodes
        nearest = find_nearest_nodes(mesh, Vec(0.5, 0.5, 0.5), 4)
        @test length(nearest) == 4
    end

    @testset "Node Operations - Find Nearest with Node Set" begin
        nodes = [
            Vec(0.0, 0.0, 0.0),  # 1
            Vec(1.0, 0.0, 0.0),  # 2
            Vec(0.0, 1.0, 0.0),  # 3
            Vec(0.0, 0.0, 1.0)   # 4
        ]
        connectivity = [(UInt32(1), UInt32(2), UInt32(3), UInt32(4))]
        node_sets = Dict(:boundary => Set(UInt32[2, 3, 4]))  # Exclude node 1
        mesh = Mesh{Tet4}(nodes, connectivity; node_sets=node_sets)

        # Find nearest in boundary set to origin
        # Node 1 is closest, but excluded, so should be node 2, 3, or 4
        nearest = find_nearest_node(mesh, Vec(0.0, 0.0, 0.0); node_set=:boundary)
        @test nearest in UInt32[2, 3, 4]
        @test nearest != UInt32(1)
    end

    @testset "Element Set Operations" begin
        nodes = [
            Vec(0.0, 0.0, 0.0), Vec(1.0, 0.0, 0.0),
            Vec(0.0, 1.0, 0.0), Vec(0.0, 0.0, 1.0),
            Vec(1.0, 1.0, 0.0)
        ]
        connectivity = [
            (UInt32(1), UInt32(2), UInt32(3), UInt32(4)),
            (UInt32(2), UInt32(5), UInt32(3), UInt32(4))
        ]
        element_sets = Dict(:all => Set(UInt32[1, 2]), :first => Set(UInt32[1]))
        mesh = Mesh{Tet4}(nodes, connectivity, element_sets)

        @test get_element_set(mesh, :all) == Set(UInt32[1, 2])
        @test get_element_set(mesh, :first) == Set(UInt32[1])

        @test get_elements_in_set(mesh, :all) == UInt32[1, 2]
        @test get_elements_in_set(mesh, :first) == UInt32[1]

        @test_throws AssertionError get_element_set(mesh, :nonexistent)
    end

    @testset "Node Set Operations" begin
        nodes = [
            Vec(0.0, 0.0, 0.0),  # 1
            Vec(1.0, 0.0, 0.0),  # 2
            Vec(0.0, 1.0, 0.0),  # 3
            Vec(0.0, 0.0, 1.0)   # 4
        ]
        connectivity = [(UInt32(1), UInt32(2), UInt32(3), UInt32(4))]
        node_sets = Dict(:all => Set(UInt32[1, 2, 3, 4]), :corner => Set(UInt32[1]))
        mesh = Mesh{Tet4}(nodes, connectivity; node_sets=node_sets)

        @test get_node_set(mesh, :all) == Set(UInt32[1, 2, 3, 4])
        @test get_node_set(mesh, :corner) == Set(UInt32[1])

        @test get_nodes_in_set(mesh, :all) == UInt32[1, 2, 3, 4]
        @test get_nodes_in_set(mesh, :corner) == UInt32[1]

        @test_throws AssertionError get_node_set(mesh, :nonexistent)
    end

    @testset "Create Node Set from Element Set" begin
        nodes = [
            Vec(0.0, 0.0, 0.0),  # 1
            Vec(1.0, 0.0, 0.0),  # 2
            Vec(0.0, 1.0, 0.0),  # 3
            Vec(0.0, 0.0, 1.0),  # 4
            Vec(1.0, 1.0, 0.0)   # 5
        ]
        connectivity = [
            (UInt32(1), UInt32(2), UInt32(3), UInt32(4)),  # Element 1 uses nodes 1,2,3,4
            (UInt32(2), UInt32(5), UInt32(3), UInt32(4))   # Element 2 uses nodes 2,5,3,4
        ]
        element_sets = Dict(:first => Set(UInt32[1]))
        mesh = Mesh{Tet4}(nodes, connectivity, element_sets)

        # Create node set from element set :first
        create_node_set_from_element_set!(mesh, :first)

        @test haskey(mesh.node_sets, :first)
        @test mesh.node_sets[:first] == Set(UInt32[1, 2, 3, 4])

        # Create with different name
        create_node_set_from_element_set!(mesh, :first, :first_nodes)
        @test haskey(mesh.node_sets, :first_nodes)
        @test mesh.node_sets[:first_nodes] == Set(UInt32[1, 2, 3, 4])
    end

    @testset "Surface Extraction - Tet4 to Tri3" begin
        nodes = [
            Vec(0.0, 0.0, 0.0),
            Vec(1.0, 0.0, 0.0),
            Vec(0.0, 1.0, 0.0),
            Vec(0.0, 0.0, 1.0)
        ]
        connectivity = [(UInt32(1), UInt32(2), UInt32(3), UInt32(4))]
        element_sets = Dict(:volume => Set(UInt32[1]))
        mesh = Mesh{Tet4}(nodes, connectivity, element_sets)

        surface = extract_surface(mesh, :volume)

        @test surface isa Mesh{Tri3}
        @test nnodes_total(surface) == 4  # Same nodes
        @test nelements(surface) == 1
        @test nnodes_per_element(surface) == 3
    end

    @testset "Surface Extraction - Tet10 to Tri6" begin
        nodes = [Vec(Float64(i - 1), 0.0, 0.0) for i in 1:10]
        connectivity = [ntuple(i -> UInt32(i), 10)]
        element_sets = Dict(:volume => Set(UInt32[1]))
        mesh = Mesh{Tet10}(nodes, connectivity, element_sets)

        surface = extract_surface(mesh, :volume)

        @test surface isa Mesh{Tri6}
        @test nnodes_total(surface) == 10  # Same nodes
        @test nelements(surface) == 1
        @test nnodes_per_element(surface) == 6
    end

    @testset "Validation - Valid Mesh" begin
        nodes = [
            Vec(0.0, 0.0, 0.0),
            Vec(1.0, 0.0, 0.0),
            Vec(0.0, 1.0, 0.0),
            Vec(0.0, 0.0, 1.0)
        ]
        connectivity = [(UInt32(1), UInt32(2), UInt32(3), UInt32(4))]
        element_sets = Dict(:all => Set(UInt32[1]))
        node_sets = Dict(:corner => Set(UInt32[1]))
        mesh = Mesh{Tet4}(nodes, connectivity, element_sets, node_sets)

        @test validate(mesh) == true
    end

    @testset "Info and Show" begin
        nodes = [
            Vec(0.0, 0.0, 0.0),
            Vec(1.0, 0.0, 0.0),
            Vec(0.0, 1.0, 0.0),
            Vec(0.0, 0.0, 1.0)
        ]
        connectivity = [(UInt32(1), UInt32(2), UInt32(3), UInt32(4))]
        element_sets = Dict(:all => Set(UInt32[1]))
        node_sets = Dict(:corner => Set(UInt32[1]))
        mesh = Mesh{Tet4}(nodes, connectivity, element_sets, node_sets)

        # Test show() produces string
        io = IOBuffer()
        show(io, mesh)
        str = String(take!(io))
        @test occursin("Mesh{Tet4}", str)
        @test occursin("4 nodes", str)
        @test occursin("1 elements", str)

        # Test info() runs without error (just call it, output goes to stdout)
        @test begin
            info(mesh)
            true
        end
    end

    @testset "Different Topology Types" begin
        # Hex8
        nodes_hex = [Vec(Float64(i - 1), 0.0, 0.0) for i in 1:8]
        connectivity_hex = [ntuple(i -> UInt32(i), 8)]
        mesh_hex = Mesh{Hex8}(nodes_hex, connectivity_hex)
        @test mesh_hex isa Mesh{Hex8}
        @test nnodes_per_element(mesh_hex) == 8

        # Seg2
        nodes_seg = [Vec(0.0, 0.0, 0.0), Vec(1.0, 0.0, 0.0)]
        connectivity_seg = [(UInt32(1), UInt32(2))]
        mesh_seg = Mesh{Seg2}(nodes_seg, connectivity_seg)
        @test mesh_seg isa Mesh{Seg2}
        @test nnodes_per_element(mesh_seg) == 2

        # Tri6
        nodes_tri = [Vec(Float64(i - 1), 0.0, 0.0) for i in 1:6]
        connectivity_tri = [ntuple(i -> UInt32(i), 6)]
        mesh_tri = Mesh{Tri6}(nodes_tri, connectivity_tri)
        @test mesh_tri isa Mesh{Tri6}
        @test nnodes_per_element(mesh_tri) == 6
    end

end

println("✅ All Mesh{T} parametric tests passed!")
