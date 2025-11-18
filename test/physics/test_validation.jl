# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Validation tests for Physics types.

Tests error handling and edge cases.
"""

using Test
using JuliaFEM
using Tensors

@testset "Physics Validation" begin
    # Helper to create minimal mesh
    function create_test_mesh()
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
        element_sets = Dict{Symbol,Set{UInt32}}(:all => Set(UInt32(1)), :solid => Set(UInt32(1)))
        return Mesh{8,Hexahedron{8}}(nodes, connectivity, element_sets)
    end

    @testset "Physics mesh reference (not copy)" begin
        mesh = create_test_mesh()
        material = LinearElastic(E=210e9, ν=0.3)

        physics1 = Physics(
            name="physics1",
            mesh=mesh,
            element_set=:all,
            field=Displacement{3}(),
            formulation=ContinuumFormulation{FullThreeD}(),
            material=material
        )

        physics2 = Physics(
            name="physics2",
            mesh=mesh,
            element_set=:solid,
            field=Displacement{3}(),
            formulation=ContinuumFormulation{FullThreeD}(),
            material=material
        )

        # Both physics reference the SAME mesh object
        @test physics1.mesh === mesh
        @test physics2.mesh === mesh
        @test physics1.mesh === physics2.mesh
    end

    @testset "Type parameter specialization" begin
        mesh = create_test_mesh()
        material = LinearElastic(E=210e9, ν=0.3)

        # 3D displacement
        physics_3d = Physics(
            name="3d",
            mesh=mesh,
            element_set=:all,
            field=Displacement{3}(),
            formulation=ContinuumFormulation{FullThreeD}(),
            material=material
        )

        # Verify type parameters can be used for dispatch
        @test physics_3d isa Physics{
            ContinuumFormulation{FullThreeD},
            Displacement{3},
            Mesh{8,Hexahedron{8}},
            LinearElastic
        }

        # Type is fully concrete
        @test isconcretetype(typeof(physics_3d))
    end

    @testset "Multiple materials with same mesh" begin
        mesh = create_test_mesh()

        material1 = LinearElastic(E=210e9, ν=0.3)
        material2 = LinearElastic(E=70e9, ν=0.33)  # Aluminum

        physics1 = Physics(
            name="steel",
            mesh=mesh,
            element_set=:all,
            field=Displacement{3}(),
            formulation=ContinuumFormulation{FullThreeD}(),
            material=material1
        )

        physics2 = Physics(
            name="aluminum",
            mesh=mesh,
            element_set=:all,
            field=Displacement{3}(),
            formulation=ContinuumFormulation{FullThreeD}(),
            material=material2
        )

        # Same mesh, different materials
        @test physics1.mesh === physics2.mesh
        @test physics1.material !== physics2.material
        @test physics1.material.E ≈ 210e9
        @test physics2.material.E ≈ 70e9
    end

    @testset "Boundary conditions independence" begin
        mesh = create_test_mesh()
        material = LinearElastic(E=210e9, ν=0.3)

        physics1 = Physics(
            name="p1",
            mesh=mesh,
            element_set=:all,
            field=Displacement{3}(),
            formulation=ContinuumFormulation{FullThreeD}(),
            material=material
        )

        physics2 = Physics(
            name="p2",
            mesh=mesh,
            element_set=:all,
            field=Displacement{3}(),
            formulation=ContinuumFormulation{FullThreeD}(),
            material=material
        )

        # Add BCs to physics1
        add_dirichlet!(physics1, [1, 2], [1, 2, 3], 0.0)
        add_neumann!(physics2, [7, 8], Vec{3}((0.0, -1000.0, 0.0)))

        # BCs are independent
        @test length(physics1.bc_dirichlet.node_ids) == 2
        @test length(physics2.bc_dirichlet.node_ids) == 0
        @test length(physics1.bc_neumann.surface_ids) == 0
        @test length(physics2.bc_neumann.surface_ids) == 2
    end
end
