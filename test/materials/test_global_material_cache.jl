# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using Test
using JuliaFEM
using Tensors

@testset "Global Material Cache" begin
    @testset "Basic construction" begin
        # Test stateless material
        mat = LinearElastic(E=210e9, ν=0.3)
        cache = create_global_material_cache(mat, n_ips=4, n_elems=10)

        @test cache isa GlobalMaterialCache
        @test size(cache.states) == (4, 10)
        @test size(cache.states_old) == (4, 10)

        # All states should be empty NamedTuple for stateless
        for elem_id in 1:10, ip in 1:4
            @test get_state(cache, ip, elem_id) === NamedTuple()
            @test get_old_state(cache, ip, elem_id) === NamedTuple()
        end
    end

    @testset "State access and update" begin
        # Create a simple stateful cache
        StateType = NamedTuple{(:κ,), Tuple{Float64}}
        cache = GlobalMaterialCache{StateType}(2, 3)  # 2 IPs, 3 elements

        # Test initial zeros
        @test get_state(cache, 1, 1).κ == 0.0
        @test get_state(cache, 2, 3).κ == 0.0

        # Set a state
        set_state!(cache, 1, 2, (κ=0.5,))
        @test get_state(cache, 1, 2).κ == 0.5
        @test get_old_state(cache, 1, 2).κ == 0.0  # Old unchanged

        # Update cache (copy current → old)
        update_cache!(cache)
        @test get_old_state(cache, 1, 2).κ == 0.5  # Now matches current

        # Modify current
        set_state!(cache, 1, 2, (κ=0.8,))
        @test get_state(cache, 1, 2).κ == 0.8
        @test get_old_state(cache, 1, 2).κ == 0.5  # Still from previous step

        # Reset
        reset_cache!(cache)
        @test get_state(cache, 1, 2).κ == 0.0
        @test get_old_state(cache, 1, 2).κ == 0.0
    end

    @testset "State variable helpers" begin
        # Create cache with plasticity-like state
        StateType = NamedTuple{(:ε_p, :α, :κ), Tuple{SymmetricTensor{2,3,Float64,6}, SymmetricTensor{2,3,Float64,6}, Float64}}
        cache = GlobalMaterialCache{StateType}(1, 1)

        # Set a complex state
        ε_p_val = SymmetricTensor{2,3}((0.001, 0.0, 0.0, 0.0, 0.0, 0.0))
        α_val = SymmetricTensor{2,3}((100e6, 0.0, 0.0, 0.0, 0.0, 0.0))
        κ_val = 0.01
        state = (ε_p=ε_p_val, α=α_val, κ=κ_val)
        set_state!(cache, 1, 1, state)

        # Get state and extract variables
        retrieved_state = get_state(cache, 1, 1)
        extracted_ε_p = get_state_variable(retrieved_state, PlasticStrain)
        extracted_α = get_state_variable(retrieved_state, Backstress)
        extracted_κ = get_state_variable(retrieved_state, EquivalentPlasticStrain)

        @test extracted_ε_p == ε_p_val
        @test extracted_α == α_val
        @test extracted_κ == κ_val

        # Test immutable update
        new_κ = 0.02
        new_state = set_state_variable(retrieved_state, EquivalentPlasticStrain, new_κ)
        @test new_state.κ == new_κ
        @test new_state.ε_p == ε_p_val  # Unchanged
        @test new_state.α == α_val      # Unchanged

        # Original unchanged (immutable)
        @test retrieved_state.κ == κ_val
    end

    @testset "Integration with traits" begin
        # Verify that cache creation uses material traits correctly
        mat = LinearElastic(E=210e9, ν=0.3)

        # Material should be stateless
        @test required_state_variables(mat) === ()

        # Cache should be empty
        cache = create_global_material_cache(mat, n_ips=2, n_elems=5)
        @test cache isa GlobalMaterialCache{NamedTuple{(),Tuple{}}}
    end
end
