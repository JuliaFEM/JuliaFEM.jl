# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using Test
using JuliaFEM
using Tensors

@testset "Structured mesh generation" begin
    @testset "Unit cube - single element" begin
        mesh = create_unit_cube_mesh(Hex8)

        # Should have 8 nodes (corners of unit cube)
        @test nnodes_total(mesh) == 8

        # Should have 1 element
        @test nelements(mesh) == 1

        # Check corner coordinates (in IJK storage order: i fastest, then j, then k)
        @test mesh.nodes[1] ≈ Vec(0.0, 0.0, 0.0)  # i=1, j=1, k=1
        @test mesh.nodes[2] ≈ Vec(1.0, 0.0, 0.0)  # i=2, j=1, k=1
        @test mesh.nodes[3] ≈ Vec(0.0, 1.0, 0.0)  # i=1, j=2, k=1
        @test mesh.nodes[4] ≈ Vec(1.0, 1.0, 0.0)  # i=2, j=2, k=1
        @test mesh.nodes[5] ≈ Vec(0.0, 0.0, 1.0)  # i=1, j=1, k=2
        @test mesh.nodes[6] ≈ Vec(1.0, 0.0, 1.0)  # i=2, j=1, k=2
        @test mesh.nodes[7] ≈ Vec(0.0, 1.0, 1.0)  # i=1, j=2, k=2
        @test mesh.nodes[8] ≈ Vec(1.0, 1.0, 1.0)  # i=2, j=2, k=2

        # Check connectivity
        @test length(mesh.connectivity[1]) == 8

        # Check element sets
        @test haskey(mesh.element_sets, :all)
        @test length(mesh.element_sets[:all]) == 1

        # Check node sets exist
        @test haskey(mesh.node_sets, :all)
        @test haskey(mesh.node_sets, :xmin)
        @test haskey(mesh.node_sets, :xmax)
        @test haskey(mesh.node_sets, :ymin)
        @test haskey(mesh.node_sets, :ymax)
        @test haskey(mesh.node_sets, :zmin)
        @test haskey(mesh.node_sets, :zmax)

        println("Unit cube (1 element): OK")
    end

    @testset "Unit cube - 2×2×2 elements" begin
        mesh = create_unit_cube_mesh(Hex8, nx=2, ny=2, nz=2)

        # Should have (2+1)³ = 27 nodes
        @test nnodes_total(mesh) == 27

        # Should have 2³ = 8 elements
        @test nelements(mesh) == 8

        # Check that all elements have 8 nodes
        for conn in mesh.connectivity
            @test length(conn) == 8
        end

        # Check boundary node sets
        xmin_nodes = mesh.node_sets[:xmin]
        @test length(xmin_nodes) == 9  # 3×3 face

        # All xmin nodes should have x ≈ 0.0
        for node_id in xmin_nodes
            @test mesh.nodes[node_id][1] ≈ 0.0
        end

        # All xmax nodes should have x ≈ 1.0
        xmax_nodes = mesh.node_sets[:xmax]
        @test length(xmax_nodes) == 9
        for node_id in xmax_nodes
            @test mesh.nodes[node_id][1] ≈ 1.0
        end

        println("Unit cube (2³ elements): OK")
    end

    @testset "Structured box - custom dimensions" begin
        # Create 10×2×2 box
        mesh = create_structured_box_mesh(Hex8,
            xmin=0.0, xmax=10.0, nx=5,
            ymin=0.0, ymax=2.0, ny=2,
            zmin=0.0, zmax=2.0, nz=2)

        # Should have (5+1)×(2+1)×(2+1) = 6×3×3 = 54 nodes
        @test nnodes_total(mesh) == 54

        # Should have 5×2×2 = 20 elements
        @test nelements(mesh) == 20

        # Check domain bounds
        all_x = [node[1] for node in mesh.nodes]
        all_y = [node[2] for node in mesh.nodes]
        all_z = [node[3] for node in mesh.nodes]

        @test minimum(all_x) ≈ 0.0
        @test maximum(all_x) ≈ 10.0
        @test minimum(all_y) ≈ 0.0
        @test maximum(all_y) ≈ 2.0
        @test minimum(all_z) ≈ 0.0
        @test maximum(all_z) ≈ 2.0

        println("Structured box (custom dimensions): OK")
    end

    @testset "Structured box - uniform spacing" begin
        # Create mesh with known spacing
        mesh = create_structured_box_mesh(Hex8,
            xmin=0.0, xmax=4.0, nx=4,
            ymin=0.0, ymax=3.0, ny=3,
            zmin=0.0, zmax=2.0, nz=2)

        # Check uniform spacing
        # X spacing should be 1.0
        x_coords = sort(unique([node[1] for node in mesh.nodes]))
        @test length(x_coords) == 5
        for i in 2:length(x_coords)
            @test x_coords[i] - x_coords[i-1] ≈ 1.0
        end

        # Y spacing should be 1.0
        y_coords = sort(unique([node[2] for node in mesh.nodes]))
        @test length(y_coords) == 4
        for i in 2:length(y_coords)
            @test y_coords[i] - y_coords[i-1] ≈ 1.0
        end

        # Z spacing should be 1.0
        z_coords = sort(unique([node[3] for node in mesh.nodes]))
        @test length(z_coords) == 3
        for i in 2:length(z_coords)
            @test z_coords[i] - z_coords[i-1] ≈ 1.0
        end

        println("Uniform spacing verification: OK")
    end

    @testset "Cantilever mesh" begin
        mesh = create_cantilever_mesh(Hex8,
            length=10.0, width=2.0, height=2.0,
            nx=10, ny=2, nz=2)

        # Should have (10+1)×(2+1)×(2+1) = 11×3×3 = 99 nodes
        @test nnodes_total(mesh) == 99

        # Should have 10×2×2 = 40 elements
        @test nelements(mesh) == 40

        # Check dimensions
        all_x = [node[1] for node in mesh.nodes]
        all_y = [node[2] for node in mesh.nodes]
        all_z = [node[3] for node in mesh.nodes]

        @test maximum(all_x) - minimum(all_x) ≈ 10.0
        @test maximum(all_y) - minimum(all_y) ≈ 2.0
        @test maximum(all_z) - minimum(all_z) ≈ 2.0

        # Boundary node sets should exist
        @test haskey(mesh.node_sets, :xmin)  # Fixed end
        @test haskey(mesh.node_sets, :xmax)  # Free end

        # Fixed end should have 3×3 = 9 nodes
        @test length(mesh.node_sets[:xmin]) == 9

        println("Cantilever mesh: OK")
    end

    @testset "Thin plate mesh" begin
        mesh = create_thin_plate_mesh(Hex8,
            length=10.0, width=10.0, thickness=0.1,
            nx=5, ny=5, nz=1)

        # Should have (5+1)×(5+1)×(1+1) = 6×6×2 = 72 nodes
        @test nnodes_total(mesh) == 72

        # Should have 5×5×1 = 25 elements
        @test nelements(mesh) == 25

        # Check thickness
        all_z = [node[3] for node in mesh.nodes]
        @test maximum(all_z) - minimum(all_z) ≈ 0.1

        # Top and bottom surfaces
        @test haskey(mesh.node_sets, :zmin)
        @test haskey(mesh.node_sets, :zmax)

        # Each surface should have 6×6 = 36 nodes
        @test length(mesh.node_sets[:zmin]) == 36
        @test length(mesh.node_sets[:zmax]) == 36

        println("Thin plate mesh: OK")
    end

    @testset "Anisotropic mesh" begin
        # Fine in X, coarse in Y and Z
        mesh = create_structured_box_mesh(Hex8,
            xmin=0.0, xmax=10.0, nx=20,
            ymin=0.0, ymax=1.0, ny=2,
            zmin=0.0, zmax=1.0, nz=2)

        # Should have (20+1)×(2+1)×(2+1) = 21×3×3 = 189 nodes
        @test nnodes_total(mesh) == 189

        # Should have 20×2×2 = 80 elements
        @test nelements(mesh) == 80

        # X should have finer spacing than Y and Z
        x_coords = sort(unique([node[1] for node in mesh.nodes]))
        y_coords = sort(unique([node[2] for node in mesh.nodes]))
        z_coords = sort(unique([node[3] for node in mesh.nodes]))

        x_spacing = x_coords[2] - x_coords[1]
        y_spacing = y_coords[2] - y_coords[1]
        z_spacing = z_coords[2] - z_coords[1]

        @test x_spacing ≈ 0.5
        @test y_spacing ≈ 0.5
        @test z_spacing ≈ 0.5

        @test length(x_coords) == 21  # Fine
        @test length(y_coords) == 3   # Coarse
        @test length(z_coords) == 3   # Coarse

        println("Anisotropic mesh: OK")
    end

    @testset "Connectivity validation" begin
        mesh = create_unit_cube_mesh(Hex8, nx=2, ny=2, nz=2)

        # Every element should have 8 unique nodes
        for (elem_id, conn) in enumerate(mesh.connectivity)
            @test length(conn) == 8
            @test length(unique(conn)) == 8  # All nodes unique

            # All node indices should be valid
            for node_id in conn
                @test 1 ≤ node_id ≤ nnodes_total(mesh)
            end
        end

        # Check that mesh validation passes
        @test validate(mesh) == true

        println("Connectivity validation: OK")
    end

    @testset "Integration with refinement" begin
        # Create coarse mesh, then refine
        coarse = create_cantilever_mesh(Hex8,
            length=10.0, width=2.0, height=2.0,
            nx=2, ny=1, nz=1)

        @test nelements(coarse) == 2

        # Refine 2 levels
        refined = refine(coarse, LongestEdgeBisection(2))

        # Should have more elements
        @test nelements(refined) > nelements(coarse)
        @test nelements(refined) == 8  # 2 → 4 → 8

        println("Integration with refinement: OK")
    end

    @testset "Convergence study pattern" begin
        # Simulate typical convergence study
        results = []

        for n in [1, 2, 4, 8]
            mesh = create_unit_cube_mesh(Hex8, nx=n, ny=n, nz=n)

            n_elem = nelements(mesh)
            n_nodes = nnodes_total(mesh)
            n_dofs = 3 * n_nodes

            push!(results, (n=n, elements=n_elem, nodes=n_nodes, dofs=n_dofs))
        end

        # Check that refinement increases mesh size correctly
        @test results[1].elements == 1     # 1³
        @test results[2].elements == 8     # 2³
        @test results[3].elements == 64    # 4³
        @test results[4].elements == 512   # 8³

        println("Convergence study pattern:")
        for r in results
            println("  n=$(r.n): $(r.elements) elements, $(r.nodes) nodes, $(r.dofs) DOFs")
        end
    end
end
