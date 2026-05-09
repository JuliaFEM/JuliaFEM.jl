# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

using Test
using JuliaFEM
using Tensors

@testset "State Variables" begin
    @testset "AbstractStateVariable" begin
        # Test that state variable types are defined
        @test PlasticStrain <: AbstractStateVariable
        @test Backstress <: AbstractStateVariable
        @test EquivalentPlasticStrain <: AbstractStateVariable
        @test DamageVariable <: AbstractStateVariable
        @test DamageEquivalentStrain <: AbstractStateVariable
        @test Eigenstrain <: AbstractStateVariable
        @test CreepStrain <: AbstractStateVariable
        @test ChabocheAlpha{1} <: AbstractStateVariable
    end

    @testset "state_variable_type trait" begin
        # Test that state_variable_type returns correct concrete types
        @test state_variable_type(PlasticStrain) === SymmetricTensor{2,3,Float64,6}
        @test state_variable_type(Backstress) === SymmetricTensor{2,3,Float64,6}
        @test state_variable_type(EquivalentPlasticStrain) === Float64
        @test state_variable_type(DamageVariable) === Float64
        @test state_variable_type(DamageEquivalentStrain) === Float64
        @test state_variable_type(Eigenstrain) === SymmetricTensor{2,3,Float64,6}
        @test state_variable_type(CreepStrain) === SymmetricTensor{2,3,Float64,6}
        @test state_variable_type(ChabocheAlpha{1}) === SymmetricTensor{2,3,Float64,6}
    end

    @testset "default_symbol trait" begin
        # Test that default_symbol returns correct symbols
        @test default_symbol(PlasticStrain) === :ε_p
        @test default_symbol(Backstress) === :α
        @test default_symbol(EquivalentPlasticStrain) === :κ
        @test default_symbol(DamageVariable) === :d
        @test default_symbol(DamageEquivalentStrain) === :κ_d
        @test default_symbol(Eigenstrain) === :ε_e
        @test default_symbol(CreepStrain) === :ε_c
        @test default_symbol(ChabocheAlpha{1}) === :α1
        @test default_symbol(ChabocheAlpha{2}) === :α2
    end

    @testset "Concrete type instantiation" begin
        # Test that we can create instances of the concrete types
        ε_p_type = state_variable_type(PlasticStrain)
        ε_p = zero(ε_p_type)
        @test ε_p isa SymmetricTensor{2,3,Float64,6}
        @test ε_p == zero(SymmetricTensor{2,3})

        α_type = state_variable_type(Backstress)
        α = zero(α_type)
        @test α isa SymmetricTensor{2,3,Float64,6}
        @test α == zero(SymmetricTensor{2,3})

        κ_type = state_variable_type(EquivalentPlasticStrain)
        κ = zero(κ_type)
        @test κ isa Float64
        @test κ == 0.0

        d_type = state_variable_type(DamageVariable)
        d = zero(d_type)
        @test d isa Float64
        @test d == 0.0
    end

    @testset "Symbol usage" begin
        # Test that symbols can be used in NamedTuples
        ε_p_sym = default_symbol(PlasticStrain)
        α_sym = default_symbol(Backstress)
        κ_sym = default_symbol(EquivalentPlasticStrain)

        state = NamedTuple{(ε_p_sym, α_sym, κ_sym)}((
            zero(SymmetricTensor{2,3}),
            zero(SymmetricTensor{2,3}),
            0.0
        ))

        @test haskey(state, :ε_p)
        @test haskey(state, :α)
        @test haskey(state, :κ)
        @test state.ε_p isa SymmetricTensor{2,3}
        @test state.α isa SymmetricTensor{2,3}
        @test state.κ isa Float64
    end

    @testset "Trait consistency" begin
        # Verify that all state variables have both traits defined
        state_vars = [
            PlasticStrain,
            Backstress,
            EquivalentPlasticStrain,
            DamageVariable,
            DamageEquivalentStrain,
            Eigenstrain,
            CreepStrain,
            ChabocheAlpha{1},
        ]

        for var in state_vars
            # Should not throw
            @test state_variable_type(var) !== nothing
            @test default_symbol(var) !== nothing

            # Symbol should be a Symbol
            @test default_symbol(var) isa Symbol

            # Type should be a concrete type
            T = state_variable_type(var)
            @test isconcretetype(T)
        end
    end
end
