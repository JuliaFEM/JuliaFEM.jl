# Test cache update functions (Phases 1a, 1b, 2)

@testset "Cache Updates" begin
    kernel = create_test_kernel()
    mesh = create_test_mesh()

    # Create caches
    N = 8      # Nodes per element (Hex8)
    NIP = 8    # Integration points (Gauss{2} for Hex8)

    geometry_cache = JuliaFEM.create_geometry_cache(N, NIP)
    element_cache = JuliaFEM.create_element_cache(mesh, kernel)
    material_cache = JuliaFEM.create_material_cache(kernel.material, NIP)

    # Test data
    elem_id = 1
    # u_global as Vec{3} for each node (or nothing for zero displacement)
    u_global = nothing
    state_old = create_material_state(kernel, mesh)
    Δt = 0.01

    @testset "Phase 1a: update_geometry_cache!" begin
        # Update geometry cache
        JuliaFEM.update_geometry_cache!(geometry_cache, element_cache, kernel, elem_id, mesh)

        # Verify X coordinates are extracted correctly
        X_expected = mesh.nodes[1:8]
        @test geometry_cache.X == X_expected

        # Verify physical gradients computed (check all in one test)
        all_gradients_nonzero = true
        for ip in 1:NIP
            for node in 1:N
                if !any(x -> abs(x) > 1e-10, geometry_cache.∇N_data[ip, node])
                    all_gradients_nonzero = false
                    break
                end
            end
        end
        @test all_gradients_nonzero

        # Verify detJ * weight is positive (check all in one test)
        all_detJ_positive = all(geometry_cache.detJ_w[ip] > 0.0 for ip in 1:NIP)
        @test all_detJ_positive

        # Test zero allocations (warm-up first)
        JuliaFEM.update_geometry_cache!(geometry_cache, element_cache, kernel, elem_id, mesh)
        allocs = @allocated JuliaFEM.update_geometry_cache!(geometry_cache, element_cache, kernel, elem_id, mesh)
        @test allocs == 0
    end

    @testset "Phase 1b: update_element_cache!" begin
        # Update element cache
        JuliaFEM.update_element_cache!(element_cache, kernel, elem_id, mesh, u_global)

        # Verify DOFs extracted
        @test element_cache.dofs == collect(1:24)

        # Verify u_buffer filled with zero displacements (check all in one test)
        all_u_zero = all(element_cache.u_buffer[node] == zero(Vec{3,Float64}) for node in 1:N)
        @test all_u_zero

        # Test zero allocations (warm-up first)
        JuliaFEM.update_element_cache!(element_cache, kernel, elem_id, mesh, u_global)
        allocs = @allocated JuliaFEM.update_element_cache!(element_cache, kernel, elem_id, mesh, u_global)
        @test allocs == 0
    end

    @testset "Phase 2: update_material_cache!" begin
        # Ensure geometry and element caches are updated first
        JuliaFEM.update_geometry_cache!(geometry_cache, element_cache, kernel, elem_id, mesh)
        JuliaFEM.update_element_cache!(element_cache, kernel, elem_id, mesh, u_global)

        # Update material cache
        JuliaFEM.update_material_cache!(material_cache, geometry_cache, kernel.material,
            element_cache, state_old, elem_id, Δt)

        # Verify material state computed at each integration point (check all in one test)
        # For stateless materials, stress/tangent are in σ and 𝔻 arrays, NOT in states
        all_stress_zero = true
        all_tangent_correct = true
        for ip in 1:NIP
            # For zero displacement, stress should be zero
            σ = material_cache.σ[ip]
            if !all(x -> abs(x) < 1e-10, σ)
                all_stress_zero = false
            end

            # Material tangent should be elasticity tensor
            C = material_cache.𝔻[ip]
            if !(C isa SymmetricTensor{4,3,Float64})
                all_tangent_correct = false
            end
        end
        @test all_stress_zero
        @test all_tangent_correct

        # Test zero allocations (warm-up first)
        JuliaFEM.update_material_cache!(material_cache, geometry_cache, kernel.material,
            element_cache, state_old, elem_id, Δt)
        allocs = @allocated JuliaFEM.update_material_cache!(material_cache, geometry_cache,
            kernel.material, element_cache,
            state_old, elem_id, Δt)
        @test allocs == 0
    end
end
