# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using Test
using JuliaFEM

@testset "Physics Category Types" begin
    @testset "Elasticity{Dim}" begin
        # Type construction
        e2d = Elasticity{2}()
        e3d = Elasticity{3}()

        @test e2d isa Elasticity{2}
        @test e3d isa Elasticity{3}
        @test e2d isa AbstractPhysics
        @test e3d isa AbstractPhysics

        # Type stability
        @test typeof(e2d) == Elasticity{2}
        @test typeof(e3d) == Elasticity{3}
    end

    @testset "Thermal{Dim}" begin
        # Type construction
        thermal_2d = Thermal{2}()
        thermal_3d = Thermal{3}()

        @test thermal_2d isa Thermal{2}
        @test thermal_3d isa Thermal{3}
        @test thermal_2d isa AbstractPhysics
        @test thermal_3d isa AbstractPhysics

        # Type stability
        @test typeof(thermal_2d) == Thermal{2}
        @test typeof(thermal_3d) == Thermal{3}
    end

    @testset "required_field_type trait" begin
        # Elasticity requires Displacement field
        @test required_field_type(Elasticity{2}()) == Displacement{2}
        @test required_field_type(Elasticity{3}()) == Displacement{3}

        # Thermal requires Temperature field (dimension-independent)
        @test required_field_type(Thermal{2}()) == Temperature
        @test required_field_type(Thermal{3}()) == Temperature

        # Type stability
        @inferred required_field_type(Elasticity{3}())
        @inferred required_field_type(Thermal{2}())
        @inferred required_field_type(Thermal{3}())
    end

    @testset "Physics type dispatch" begin
        # Different dimensions dispatch to different field types
        function get_field_dim(physics::Elasticity{Dim}) where Dim
            return Dim
        end

        @test get_field_dim(Elasticity{2}()) == 2
        @test get_field_dim(Elasticity{3}()) == 3

        # Thermal dimension parameter
        @test required_field_type(Thermal{2}()) isa Type{Temperature}
        @test required_field_type(Thermal{3}()) isa Type{Temperature}
    end
end
