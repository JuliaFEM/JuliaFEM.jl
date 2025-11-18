# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Unit tests for boundary condition methods.

Tests add_dirichlet! and add_neumann! implementations.
"""

using Test
using JuliaFEM
using Tensors

@testset "Boundary Conditions" begin
    # Helper to create minimal physics
    function create_test_physics()
        nodes = Vec{3,Float64}[
            Vec{3}((0.0, 0.0, 0.0)),
            Vec{3}((1.0, 0.0, 0.0)),
            Vec{3}((1.0, 1.0, 0.0)),
            Vec{3}((0.0, 1.0, 0.0)),
            Vec{3}((0.0, 0.0, 1.0)),
            Vec{3}((1.0, 0.0, 1.0)),
            Vec{3}((1.0, 1.0, 1.0)),
            Vec{3}((0.0, 1.0, 1.0))
        ]
        connectivity = [NTuple{8,UInt32}((1, 2, 3, 4, 5, 6, 7, 8))]
        element_sets = Dict{Symbol,Set{UInt32}}(:all => Set(UInt32(1)))
        mesh = Mesh{8,Hexahedron{8}}(nodes, connectivity, element_sets)
        material = LinearElastic(E=210e9, ν=0.3)

        return Physics(
            name="test",
            mesh=mesh,
            element_set=:all,
            field=Displacement{3}(),
            formulation=ContinuumFormulation{FullThreeD}(),
            material=material
        )
    end

    @testset "add_dirichlet! - single node" begin
        physics = create_test_physics()

        add_dirichlet!(physics, [1], [1, 2, 3], 0.0)

        bc = physics.bc_dirichlet
        @test length(bc.node_ids) == 1
        @test bc.node_ids[1] == 1
        @test bc.components[1] == [1, 2, 3]
        @test bc.values[1] == 0.0
    end

    @testset "add_dirichlet! - multiple nodes" begin
        physics = create_test_physics()

        add_dirichlet!(physics, [1, 2, 3], [1, 2, 3], 0.0)

        bc = physics.bc_dirichlet
        @test length(bc.node_ids) == 3
        @test bc.node_ids == [1, 2, 3]
        @test all(c == [1, 2, 3] for c in bc.components)
        @test all(v == 0.0 for v in bc.values)
    end

    @testset "add_dirichlet! - partial DOFs" begin
        physics = create_test_physics()

        add_dirichlet!(physics, [5], [3], 0.1)

        bc = physics.bc_dirichlet
        @test length(bc.node_ids) == 1
        @test bc.node_ids[1] == 5
        @test bc.components[1] == [3]
        @test bc.values[1] == 0.1
    end

    @testset "add_dirichlet! - accumulation" begin
        physics = create_test_physics()

        add_dirichlet!(physics, [1], [1, 2, 3], 0.0)
        add_dirichlet!(physics, [2], [3], 0.1)

        bc = physics.bc_dirichlet
        @test length(bc.node_ids) == 2
        @test bc.node_ids == [1, 2]
        @test bc.components == [[1, 2, 3], [3]]
        @test bc.values == [0.0, 0.1]
    end

    @testset "add_neumann! - single surface" begin
        physics = create_test_physics()

        traction = Vec{3}((0.0, -1000.0, 0.0))
        add_neumann!(physics, [1], traction)

        bc = physics.bc_neumann
        @test length(bc.surface_ids) == 1
        @test bc.surface_ids[1] == 1
        @test bc.values[1] == traction
    end

    @testset "add_neumann! - multiple surfaces" begin
        physics = create_test_physics()

        traction = Vec{3}((0.0, -1000.0, 0.0))
        add_neumann!(physics, [1, 2, 3], traction)

        bc = physics.bc_neumann
        @test length(bc.surface_ids) == 3
        @test bc.surface_ids == [1, 2, 3]
        @test all(v == traction for v in bc.values)
    end

    @testset "add_neumann! - different tractions" begin
        physics = create_test_physics()

        traction1 = Vec{3}((0.0, -1000.0, 0.0))
        traction2 = Vec{3}((100.0, 0.0, 0.0))

        add_neumann!(physics, [1], traction1)
        add_neumann!(physics, [2], traction2)

        bc = physics.bc_neumann
        @test length(bc.surface_ids) == 2
        @test bc.surface_ids == [1, 2]
        @test bc.values[1] == traction1
        @test bc.values[2] == traction2
    end

    @testset "Combined Dirichlet and Neumann BCs" begin
        physics = create_test_physics()

        # Fix some nodes
        add_dirichlet!(physics, [1, 2], [1, 2, 3], 0.0)

        # Apply traction
        traction = Vec{3}((0.0, -1000.0, 0.0))
        add_neumann!(physics, [7, 8], traction)

        # Check both BCs are stored
        @test length(physics.bc_dirichlet.node_ids) == 2
        @test length(physics.bc_neumann.surface_ids) == 2

        # Verify independence
        @test physics.bc_dirichlet.node_ids == [1, 2]
        @test physics.bc_neumann.surface_ids == [7, 8]
    end
end
