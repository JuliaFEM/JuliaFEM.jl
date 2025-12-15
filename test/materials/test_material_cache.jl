# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using Test
using JuliaFEM
using Tensors

@testset "Material Cache" begin
    @testset "State type inference" begin
        # Test that material_state_type correctly infers from traits

        # Stateless material (LinearElastic)
        mat_stateless = LinearElastic(E=210e9, ν=0.3)
        StateType_stateless = JuliaFEM.material_state_type(mat_stateless)
        @test StateType_stateless === NamedTuple{(),Tuple{}}
        @test fieldnames(StateType_stateless) == ()

        # Stateful material (PerfectPlasticity)
        # Note: PerfectPlasticity not exported yet, so this test will be added later
        # when PerfectPlasticity is integrated
    end

    @testset "Zero state creation" begin
        # Test empty state
        StateType_empty = NamedTuple{(),Tuple{}}
        zero_state = JuliaFEM.create_zero_state(StateType_empty)
        @test zero_state === NamedTuple()

        # Test tensor state
        StateType_tensor = NamedTuple{(:ε_p,), Tuple{SymmetricTensor{2,3,Float64,6}}}
        zero_state = JuliaFEM.create_zero_state(StateType_tensor)
        @test haskey(zero_state, :ε_p)
        @test zero_state.ε_p == zero(SymmetricTensor{2,3})

        # Test scalar state
        StateType_scalar = NamedTuple{(:κ,), Tuple{Float64}}
        zero_state = JuliaFEM.create_zero_state(StateType_scalar)
        @test haskey(zero_state, :κ)
        @test zero_state.κ == 0.0

        # Test mixed state (like plasticity)
        StateType_mixed = NamedTuple{(:ε_p, :α, :κ), Tuple{SymmetricTensor{2,3,Float64,6}, SymmetricTensor{2,3,Float64,6}, Float64}}
        zero_state = JuliaFEM.create_zero_state(StateType_mixed)
        @test haskey(zero_state, :ε_p)
        @test haskey(zero_state, :α)
        @test haskey(zero_state, :κ)
        @test zero_state.ε_p == zero(SymmetricTensor{2,3})
        @test zero_state.α == zero(SymmetricTensor{2,3})
        @test zero_state.κ == 0.0
    end

    @testset "MaterialCache construction - stateless" begin
        mat = LinearElastic(E=210e9, ν=0.3)
        n_ips = 8

        cache = JuliaFEM.create_global_material_cache(mat, n_ips=n_ips, n_elems=10)

        # Check type
        @test cache isa JuliaFEM.GlobalMaterialCache{NamedTuple{(),Tuple{}}}

        # Check structure (Matrix: [nips, nelems])
        @test size(cache.states) == (n_ips, 10)
        @test size(cache.states_old) == (n_ips, 10)

        # Check initialization
        for elem_id in 1:10, ip in 1:n_ips
            @test cache.states[ip, elem_id] === NamedTuple()
            @test cache.states_old[ip, elem_id] === NamedTuple()
        end
    end

    @testset "MaterialCache construction - stateful" begin
        # Define a simple stateful state type manually
        StateType = NamedTuple{(:ε_p, :κ), Tuple{SymmetricTensor{2,3,Float64,6}, Float64}}
        n_ips = 4
        n_elems = 3

        cache = JuliaFEM.GlobalMaterialCache{StateType}(n_ips, n_elems)

        # Check type
        @test cache isa JuliaFEM.GlobalMaterialCache{StateType}

        # Check structure (Matrix: [nips, nelems])
        @test size(cache.states) == (n_ips, n_elems)
        @test size(cache.states_old) == (n_ips, n_elems)

        # Check initialization to zeros
        for elem_id in 1:n_elems, ip in 1:n_ips
            state = cache.states[ip, elem_id]
            @test haskey(state, :ε_p)
            @test haskey(state, :κ)
            @test state.ε_p == zero(SymmetricTensor{2,3})
            @test state.κ == 0.0

            state_old = cache.states_old[ip, elem_id]
            @test haskey(state_old, :ε_p)
            @test haskey(state_old, :κ)
            @test state_old.ε_p == zero(SymmetricTensor{2,3})
            @test state_old.κ == 0.0
        end
    end

    @testset "State access functions" begin
        StateType = NamedTuple{(:ε_p, :κ), Tuple{SymmetricTensor{2,3,Float64,6}, Float64}}
        n_ips = 4
        n_elems = 2
        cache = JuliaFEM.GlobalMaterialCache{StateType}(n_ips, n_elems)

        # Test get_state (requires ip and elem_id)
        ip = 2
        elem_id = 1
        state = JuliaFEM.get_state(cache, ip, elem_id)
        @test state isa NamedTuple
        @test haskey(state, :ε_p)
        @test haskey(state, :κ)

        # Test get_old_state
        state_old = JuliaFEM.get_old_state(cache, ip, elem_id)
        @test state_old isa NamedTuple
        @test haskey(state_old, :ε_p)
        @test haskey(state_old, :κ)

        # Test set_state!
        new_ε_p = SymmetricTensor{2,3}((0.001, 0.0, 0.0, 0.0, 0.0, 0.0))
        new_κ = 0.5
        new_state = (ε_p=new_ε_p, κ=new_κ)

        JuliaFEM.set_state!(cache, ip, elem_id, new_state)

        state_updated = JuliaFEM.get_state(cache, ip, elem_id)
        @test state_updated.ε_p == new_ε_p
        @test state_updated.κ == new_κ

        # Verify old state unchanged
        state_old_check = JuliaFEM.get_old_state(cache, ip, elem_id)
        @test state_old_check.ε_p == zero(SymmetricTensor{2,3})
        @test state_old_check.κ == 0.0
    end

    @testset "update_cache! - time stepping" begin
        StateType = NamedTuple{(:κ,), Tuple{Float64}}
        n_ips = 3
        n_elems = 1
        cache = JuliaFEM.GlobalMaterialCache{StateType}(n_ips, n_elems)
        elem_id = 1

        # Set some current states
        JuliaFEM.set_state!(cache, 1, elem_id, (κ=0.1,))
        JuliaFEM.set_state!(cache, 2, elem_id, (κ=0.2,))
        JuliaFEM.set_state!(cache, 3, elem_id, (κ=0.3,))

        # Old states should still be zero
        @test JuliaFEM.get_old_state(cache, 1, elem_id).κ == 0.0
        @test JuliaFEM.get_old_state(cache, 2, elem_id).κ == 0.0
        @test JuliaFEM.get_old_state(cache, 3, elem_id).κ == 0.0

        # Update cache (copy current → old)
        JuliaFEM.update_cache!(cache)

        # Now old states should match current
        @test JuliaFEM.get_old_state(cache, 1, elem_id).κ == 0.1
        @test JuliaFEM.get_old_state(cache, 2, elem_id).κ == 0.2
        @test JuliaFEM.get_old_state(cache, 3, elem_id).κ == 0.3

        # Current states unchanged
        @test JuliaFEM.get_state(cache, 1, elem_id).κ == 0.1
        @test JuliaFEM.get_state(cache, 2, elem_id).κ == 0.2
        @test JuliaFEM.get_state(cache, 3, elem_id).κ == 0.3

        # Modify current states
        JuliaFEM.set_state!(cache, 1, elem_id, (κ=0.15,))
        JuliaFEM.set_state!(cache, 2, elem_id, (κ=0.25,))

        # Old states should still be from previous step
        @test JuliaFEM.get_old_state(cache, 1, elem_id).κ == 0.1
        @test JuliaFEM.get_old_state(cache, 2, elem_id).κ == 0.2

        # Current states are new
        @test JuliaFEM.get_state(cache, 1, elem_id).κ == 0.15
        @test JuliaFEM.get_state(cache, 2, elem_id).κ == 0.25
    end

    @testset "reset_cache!" begin
        StateType = NamedTuple{(:ε_p, :κ), Tuple{SymmetricTensor{2,3,Float64,6}, Float64}}
        n_ips = 2
        n_elems = 1
        cache = JuliaFEM.GlobalMaterialCache{StateType}(n_ips, n_elems)
        elem_id = 1

        # Set some non-zero states
        ε_p = SymmetricTensor{2,3}((0.01, 0.0, 0.0, 0.0, 0.0, 0.0))
        JuliaFEM.set_state!(cache, 1, elem_id, (ε_p=ε_p, κ=0.5))
        JuliaFEM.set_state!(cache, 2, elem_id, (ε_p=ε_p, κ=0.8))

        # Update to move to old
        JuliaFEM.update_cache!(cache)

        # Verify non-zero
        @test JuliaFEM.get_state(cache, 1, elem_id).κ != 0.0
        @test JuliaFEM.get_old_state(cache, 1, elem_id).κ != 0.0

        # Reset
        JuliaFEM.reset_cache!(cache)

        # All should be zero
        for ip in 1:n_ips
            state = JuliaFEM.get_state(cache, ip, elem_id)
            state_old = JuliaFEM.get_old_state(cache, ip, elem_id)

            @test state.ε_p == zero(SymmetricTensor{2,3})
            @test state.κ == 0.0
            @test state_old.ε_p == zero(SymmetricTensor{2,3})
            @test state_old.κ == 0.0
        end
    end

    @testset "State variable helpers" begin
        # Create a state matching plasticity structure
        StateType = NamedTuple{(:ε_p, :α, :κ), Tuple{SymmetricTensor{2,3,Float64,6}, SymmetricTensor{2,3,Float64,6}, Float64}}

        ε_p_val = SymmetricTensor{2,3}((0.001, 0.0, 0.0, 0.0, 0.0, 0.0))
        α_val = SymmetricTensor{2,3}((100e6, 0.0, 0.0, 0.0, 0.0, 0.0))
        κ_val = 0.01

        state = (ε_p=ε_p_val, α=α_val, κ=κ_val)

        # Test get_state_variable
        extracted_ε_p = JuliaFEM.get_state_variable(state, PlasticStrain)
        @test extracted_ε_p == ε_p_val

        extracted_α = JuliaFEM.get_state_variable(state, Backstress)
        @test extracted_α == α_val

        extracted_κ = JuliaFEM.get_state_variable(state, EquivalentPlasticStrain)
        @test extracted_κ == κ_val

        # Test set_state_variable (immutable update)
        new_κ_val = 0.02
        new_state = JuliaFEM.set_state_variable(state, EquivalentPlasticStrain, new_κ_val)

        # Original state unchanged (immutable)
        @test state.κ == κ_val

        # New state has updated value
        @test new_state.κ == new_κ_val

        # Other values unchanged
        @test new_state.ε_p == ε_p_val
        @test new_state.α == α_val
    end

    @testset "Type stability" begin
        # Verify that cache operations are type-stable
        mat = LinearElastic(E=210e9, ν=0.3)
        cache = JuliaFEM.create_global_material_cache(mat, n_ips=4, n_elems=2)

        # get_state should be type-stable (requires ip and elem_id)
        @inferred JuliaFEM.get_state(cache, 1, 1)
        @inferred JuliaFEM.get_old_state(cache, 1, 1)

        # For stateful cache
        StateType = NamedTuple{(:κ,), Tuple{Float64}}
        cache_stateful = JuliaFEM.GlobalMaterialCache{StateType}(4, 2)

        @inferred JuliaFEM.get_state(cache_stateful, 1, 1)
        @inferred JuliaFEM.get_old_state(cache_stateful, 1, 1)
    end

    @testset "Integration with material traits" begin
        # Verify that the cache system integrates correctly with material traits
        mat = LinearElastic(E=210e9, ν=0.3)

        # Material declares no state variables
        vars = required_state_variables(mat)
        @test vars === ()

        # Cache should be empty
        StateType = JuliaFEM.material_state_type(mat)
        @test StateType === NamedTuple{(),Tuple{}}

        # Created cache should be stateless
        cache = JuliaFEM.create_global_material_cache(mat, n_ips=2, n_elems=3)
        @test cache isa JuliaFEM.GlobalMaterialCache{NamedTuple{(),Tuple{}}}
        @test JuliaFEM.get_state(cache, 1, 1) === NamedTuple()
    end
end
