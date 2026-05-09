# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

using Test
using JuliaFEM
using Tensors

@testset "PerfectPlasticity Integration" begin
    @testset "Material instantiation and traits" begin
        # Create material
        mat = PerfectPlasticity(E=210e9, ν=0.3, σ_y=250e6, H=1e9)

        @test mat isa PerfectPlasticity
        @test mat.E == 210e9
        @test mat.ν == 0.3
        @test mat.σ_y == 250e6
        @test mat.H == 1e9

        # Test material traits
        physics = supported_physics(mat)
        @test physics isa Tuple
        @test length(physics) == 1
        @test physics[1] isa Elasticity{3}

        # Test state variable requirements
        state_vars = required_state_variables(mat)
        @test state_vars == (PlasticStrain, Backstress, EquivalentPlasticStrain)

        # Material should be stateful
        @test is_stateful(mat)
    end

    @testset "Global material cache creation" begin
        mat = PerfectPlasticity(E=210e9, ν=0.3, σ_y=250e6, H=1e9)
        n_ips = 8
        n_elems = 10

        # Create cache automatically from material
        cache = create_global_material_cache(mat, n_ips=n_ips, n_elems=n_elems)

        @test cache isa GlobalMaterialCache
        @test size(cache.states) == (n_ips, n_elems)
        @test size(cache.states_old) == (n_ips, n_elems)

        # All states should be initialized to zero
        for elem_id in 1:n_elems
            for ip in 1:n_ips
                state = get_state(cache, ip, elem_id)
                @test haskey(state, :ε_p)
                @test haskey(state, :α)
                @test haskey(state, :κ)
                @test state.ε_p == zero(SymmetricTensor{2,3})
                @test state.α == zero(SymmetricTensor{2,3})
                @test state.κ == 0.0
            end
        end
    end

    @testset "Direct NamedTuple state manipulation" begin
        # Test compositional state creation and access
        state_nt = (
            ε_p=SymmetricTensor{2,3}((0.001, 0.0, 0.0, 0.0, 0.0, 0.0)),
            α=SymmetricTensor{2,3}((100e6, 0.0, 0.0, 0.0, 0.0, 0.0)),
            κ=0.01
        )

        # State is just a NamedTuple - no conversion needed!
        @test state_nt isa NamedTuple
        @test haskey(state_nt, :ε_p)
        @test haskey(state_nt, :α)
        @test haskey(state_nt, :κ)

        # Can use directly with compute_stress
        mat = PerfectPlasticity(E=210e9, ν=0.3, σ_y=250e6, H=1e9)
        ε = SymmetricTensor{2,3}((0.002, 0.0, 0.0, 0.0, 0.0, 0.0))
        σ, 𝔻, state_new = compute_stress(mat, ε, state_nt, 0.0)

        @test state_new isa NamedTuple
        @test haskey(state_new, :ε_p)
        @test haskey(state_new, :α)
        @test haskey(state_new, :κ)
    end

    @testset "Direct cache access and manipulation" begin
        mat = PerfectPlasticity(E=210e9, ν=0.3, σ_y=250e6, H=1e9)
        cache = create_global_material_cache(mat, n_ips=4, n_elems=2)

        # Set states directly - no conversion needed
        state1 = (
            ε_p=SymmetricTensor{2,3}((0.001, 0.0, 0.0, 0.0, 0.0, 0.0)),
            α=SymmetricTensor{2,3}((50e6, 0.0, 0.0, 0.0, 0.0, 0.0)),
            κ=0.005
        )
        set_state!(cache, 1, 1, state1)

        # Retrieve states directly
        retrieved_state = get_state(cache, 1, 1)
        @test retrieved_state.ε_p == state1.ε_p
        @test retrieved_state.α == state1.α
        @test retrieved_state.κ == state1.κ

        # Modify state directly
        state2 = (
            ε_p=SymmetricTensor{2,3}((0.002, 0.0, 0.0, 0.0, 0.0, 0.0)),
            α=SymmetricTensor{2,3}((100e6, 0.0, 0.0, 0.0, 0.0, 0.0)),
            κ=0.01
        )
        set_state!(cache, 2, 1, state2)

        # Verify stored correctly
        state2_retrieved = get_state(cache, 2, 1)
        @test state2_retrieved.ε_p ≈ SymmetricTensor{2,3}((0.002, 0.0, 0.0, 0.0, 0.0, 0.0))
        @test state2_retrieved.α ≈ SymmetricTensor{2,3}((100e6, 0.0, 0.0, 0.0, 0.0, 0.0))
        @test state2_retrieved.κ ≈ 0.01
    end

    @testset "Elastic response (below yield)" begin
        # Test that material behaves elastically below yield
        mat = PerfectPlasticity(E=210e9, ν=0.3, σ_y=250e6, H=1e9)

        # Small elastic strain
        ε = SymmetricTensor{2,3}((1e-4, 0.0, 0.0, 0.0, 0.0, 0.0))  # ~21 MPa << 250 MPa
        state_old = NamedTuple()  # Empty state = initial

        # Compute stress
        σ, 𝔻, state_new = compute_stress(mat, ε, state_old, 0.0)

        # Should be purely elastic (no plastic strain)
        @test state_new.ε_p == zero(SymmetricTensor{2,3})
        @test state_new.α == zero(SymmetricTensor{2,3})
        @test state_new.κ == 0.0

        # Stress should be elastic prediction
        @test norm(σ) > 0.0
        @test norm(σ) < mat.σ_y  # Below yield
    end

    @testset "J2 yield surface after radial return (‖dev(σ−α)‖_vm ≈ σᵧ)" begin
        mat = PerfectPlasticity(E = 210e9, ν = 0.3, σ_y = 250e6, H = 2.1e9)
        ε = SymmetricTensor{2,3}((0.02, -0.005, 0.0, 0.0, 0.0, 0.0))
        σ, _, st = compute_stress(mat, ε, NamedTuple(), 0.0)
        s = dev(σ - st.α)
        seq = sqrt(3 / 2 * (s ⊡ s))
        @test seq ≈ mat.σ_y rtol = 1e-5 atol = 1.0
    end

    @testset "Plastic response (above yield)" begin
        # Test that material yields when stress exceeds yield
        mat = PerfectPlasticity(E=210e9, ν=0.3, σ_y=250e6, H=1e9)

        # Large elastic strain that will cause yielding
        # σ_y = 250 MPa, E=210 GPa → ε ≈ 0.0012 should yield
        ε = SymmetricTensor{2,3}((0.005, 0.0, 0.0, 0.0, 0.0, 0.0))  # Large strain
        state_old = NamedTuple()  # Empty state = initial

        # Compute stress
        σ, 𝔻, state_new = compute_stress(mat, ε, state_old, 0.0)

        # Should have plastic strain
        @test norm(state_new.ε_p) > 0.0
        @test state_new.κ > 0.0

        # Stress should be at/near yield surface
        @test norm(σ) > 0.0
    end

    @testset "Time stepping workflow" begin
        # Simulate a complete time-stepping workflow
        mat = PerfectPlasticity(E=210e9, ν=0.3, σ_y=250e6, H=1e9)
        cache = create_global_material_cache(mat, n_ips=4, n_elems=1)

        # Step 1: Apply small load (elastic)
        ε1 = SymmetricTensor{2,3}((1e-4, 0.0, 0.0, 0.0, 0.0, 0.0))
        state1_old = get_state(cache, 1, 1)  # Get directly from cache
        σ1, 𝔻1, state1_new = compute_stress(mat, ε1, state1_old, 0.0)
        set_state!(cache, 1, 1, state1_new)  # Store directly in cache

        # Verify elastic (no plastic strain)
        @test get_state(cache, 1, 1).κ == 0.0

        # Update cache for next step
        update_cache!(cache)

        # Step 2: Apply larger load (plastic)
        ε2 = SymmetricTensor{2,3}((0.005, 0.0, 0.0, 0.0, 0.0, 0.0))
        state2_old = get_state(cache, 1, 1)  # Get directly from cache
        σ2, 𝔻2, state2_new = compute_stress(mat, ε2, state2_old, 0.0)
        set_state!(cache, 1, 1, state2_new)  # Store directly in cache

        # Verify plastic strain accumulated
        @test get_state(cache, 1, 1).κ > 0.0

        # Old state should still be from step 1 (elastic)
        @test get_old_state(cache, 1, 1).κ == 0.0
    end

    @testset "State variable helpers" begin
        # Test get/set state variable helpers
        state = (
            ε_p=SymmetricTensor{2,3}((0.001, 0.0, 0.0, 0.0, 0.0, 0.0)),
            α=SymmetricTensor{2,3}((100e6, 0.0, 0.0, 0.0, 0.0, 0.0)),
            κ=0.01
        )

        # Get individual variables
        ε_p = get_state_variable(state, PlasticStrain)
        α = get_state_variable(state, Backstress)
        κ = get_state_variable(state, EquivalentPlasticStrain)

        @test ε_p == state.ε_p
        @test α == state.α
        @test κ == state.κ

        # Set individual variables (immutable update)
        new_κ = 0.02
        state_updated = set_state_variable(state, EquivalentPlasticStrain, new_κ)

        @test state_updated.κ == new_κ
        @test state_updated.ε_p == state.ε_p  # Unchanged
        @test state_updated.α == state.α      # Unchanged
        @test state.κ == 0.01                  # Original unchanged
    end

    @testset "Type stability" begin
        # Verify type stability of key operations
        mat = PerfectPlasticity(E=210e9, ν=0.3, σ_y=250e6, H=1e9)
        cache = create_global_material_cache(mat, n_ips=4, n_elems=2)

        # Cache access should be type-stable
        @inferred get_state(cache, 1, 1)
        @inferred get_old_state(cache, 1, 1)

        # compute_stress should be type-stable
        state_nt = (ε_p=zero(SymmetricTensor{2,3}), α=zero(SymmetricTensor{2,3}), κ=0.0)
        ε = SymmetricTensor{2,3}((1e-4, 0.0, 0.0, 0.0, 0.0, 0.0))
        @inferred compute_stress(mat, ε, state_nt, 0.0)
    end
end
