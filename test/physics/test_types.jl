# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Unit tests for Physics types.

Tests the Physics struct and boundary condition storage types.
"""

using Test
using JuliaFEM
using Tensors

@testset "Physics Types" begin
    @testset "DirichletBC construction" begin
        bc = DirichletBC()
        @test isempty(bc.node_ids)
        @test isempty(bc.components)
        @test isempty(bc.values)
    end

    @testset "NeumannBC construction" begin
        bc = NeumannBC()
        @test isempty(bc.surface_ids)
        @test isempty(bc.values)
    end

    @testset "Constraint construction" begin
        c = Constraint()
        @test c isa Constraint
    end

    @testset "Physics construction with minimal mesh" begin
        # Create minimal 1-element Hex8 mesh
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

        # Create material
        material = LinearElastic(E=210e9, ν=0.3)

        # Create physics
        physics = Physics(
            name="test",
            mesh=mesh,
            element_set=:all,
            field=Displacement{3}(),
            formulation=ContinuumFormulation{FullThreeD}(),
            material=material
        )

        @test physics isa Physics
        @test physics.name == "test"
        @test physics.mesh === mesh  # Same reference, not copy
        @test physics.element_set == :all
        @test physics.field isa Displacement{3}
        @test physics.formulation isa ContinuumFormulation{FullThreeD}
        @test physics.material === material
        @test isempty(physics.constraints)
        @test physics.bc_dirichlet isa DirichletBC
        @test physics.bc_neumann isa NeumannBC
    end

    @testset "Physics type parameters" begin
        # Create minimal mesh
        nodes = Vec{3,Float64}[Vec{3}((0.0, 0.0, 0.0)), Vec{3}((1.0, 0.0, 0.0))]
        connectivity = [NTuple{2,UInt32}((1, 2))]
        element_sets = Dict{Symbol,Set{UInt32}}(:all => Set(UInt32(1)))
        mesh = Mesh{2,Segment{2}}(nodes, connectivity, element_sets)

        material = LinearElastic(E=210e9, ν=0.3)

        physics = Physics(
            name="test",
            mesh=mesh,
            element_set=:all,
            field=Displacement{3}(),
            formulation=ContinuumFormulation{FullThreeD}(),
            material=material
        )

        # Check type parameters are correctly inferred
        @test physics isa Physics{
            ContinuumFormulation{FullThreeD},
            Displacement{3},
            Mesh{2,Segment{2}},
            LinearElastic
        }
    end
end
