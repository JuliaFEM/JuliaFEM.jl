# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using Test
using JuliaFEM
using Tensors

@testset "Material Traits" begin
    @testset "LinearElastic traits" begin
        mat = LinearElastic(E=210e9, ν=0.3)

        # Test supported_physics
        physics = supported_physics(mat)
        @test physics isa Tuple
        @test length(physics) == 1
        @test physics[1] isa Elasticity{3}

        # Test required_field_types (derived from supported_physics)
        fields = required_field_types(mat)
        @test fields isa Tuple
        @test length(fields) == 1
        @test fields[1] === Displacement{3}

        # Test required_state_variables (stateless)
        state_vars = required_state_variables(mat)
        @test state_vars isa Tuple
        @test length(state_vars) == 0
        @test isempty(state_vars)

        # Test helper functions
        @test is_stateful(mat) == false

        state_types = get_state_variable_types(mat)
        @test state_types isa Tuple
        @test length(state_types) == 0

        state_symbols = get_state_variable_symbols(mat)
        @test state_symbols isa Tuple
        @test length(state_symbols) == 0
    end

    @testset "Trait function existence" begin
        # Test that trait functions are exported and callable
        @test isdefined(JuliaFEM, :supported_physics)
        @test isdefined(JuliaFEM, :required_field_types)
        @test isdefined(JuliaFEM, :required_state_variables)
        @test isdefined(JuliaFEM, :get_state_variable_types)
        @test isdefined(JuliaFEM, :get_state_variable_symbols)
        @test isdefined(JuliaFEM, :is_stateful)
    end

    @testset "Helper function consistency" begin
        mat = LinearElastic(E=210e9, ν=0.3)

        # For stateless material
        @test is_stateful(mat) == false
        @test isempty(required_state_variables(mat))
        @test isempty(get_state_variable_types(mat))
        @test isempty(get_state_variable_symbols(mat))
    end

    @testset "Compositional design verification" begin
        # This test verifies the compositional design philosophy:
        # Materials declare tuples of state variable types,
        # and each state variable has its own traits

        mat = LinearElastic(E=210e9, ν=0.3)

        # LinearElastic should have no state variables
        vars = required_state_variables(mat)
        @test vars === ()

        # If a material HAD state variables (e.g., PerfectPlasticity),
        # we could query each variable's traits like this:
        # vars = required_state_variables(material)
        # for var in vars
        #     concrete_type = state_variable_type(var)
        #     symbol = default_symbol(var)
        # end

        # The key insight: state variables are building blocks,
        # materials compose them to declare requirements
    end

    @testset "Physics to field type derivation" begin
        mat = LinearElastic(E=210e9, ν=0.3)

        # Test that required_field_types correctly derives from supported_physics
        physics = supported_physics(mat)
        fields = required_field_types(mat)

        @test length(physics) == length(fields)

        for (p, f) in zip(physics, fields)
            # Each physics should map to its required field
            @test f === required_field_type(p)
        end
    end
end
