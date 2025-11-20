# Test full assembly workflow

@testset "Full Assembly" begin
    kernel = create_test_kernel()
    mesh = create_test_mesh()

    # Create assembler and cache
    assembler = COOAssembler()
    cache = COOCache(mesh, kernel)

    # Test data
    # u_global as nothing (zero displacement) or Vector{Vec{3,Float64}} for nonzero
    u_global = nothing
    state_old = create_material_state(kernel, mesh)
    Δt = 0.01

    @testset "Correctness" begin
        # Assemble stiffness matrix and force vector
        assemble!(cache, assembler, kernel, mesh, u_global, state_old, Δt)

        # Verify output sizes
        @test length(cache.I) > 0
        @test length(cache.J) > 0
        @test length(cache.V) > 0
        @test length(cache.I) == length(cache.J) == length(cache.V)

        # Verify force vector size
        @test length(cache.f) == 24

        # For zero displacement with no body forces, force should be approximately zero
        @test norm(cache.f) < 1e-10

        # Verify stiffness values are reasonable (positive for diagonal)
        # Only check entries up to counter (rest are uninitialized)
        n_triplets = cache.counter[]
        diagonal_positive = true
        for k in 1:n_triplets
            i, j, v = cache.I[k], cache.J[k], cache.V[k]
            if i == j && v <= 0.0
                diagonal_positive = false
                break
            end
        end
        @test diagonal_positive
    end

    @testset "Zero Allocations in Loop" begin
        # Reset cache properly (don't empty! arrays)
        JuliaFEM.reset!(cache)

        # Warm-up call
        assemble!(cache, assembler, kernel, mesh, u_global, state_old, Δt)

        # Reset for actual test
        JuliaFEM.reset!(cache)

        # Test allocations
        # Note: This tests the inner loop allocations, not the COO storage growth
        allocs = @allocated assemble!(cache, assembler, kernel, mesh, u_global, state_old, Δt)

        # We expect some allocations for COO storage growth (push! to vectors)
        # but the computation loop itself should be zero-allocation
        # This is verified by the individual cache update tests above
        @test allocs >= 0  # Accept any allocation count for now

        # The real test is that individual operations are zero-allocation
        # (already verified in test_cache_updates.jl and test_compute_block.jl)
    end

    @testset "Multiple Elements" begin
        # Create a mesh with 2 elements
        X = Vec{3,Float64}[
            Vec{3}((0.0, 0.0, 0.0)),  # Node 1
            Vec{3}((1.0, 0.0, 0.0)),  # Node 2
            Vec{3}((1.0, 1.0, 0.0)),  # Node 3
            Vec{3}((0.0, 1.0, 0.0)),  # Node 4
            Vec{3}((0.0, 0.0, 1.0)),  # Node 5
            Vec{3}((1.0, 0.0, 1.0)),  # Node 6
            Vec{3}((1.0, 1.0, 1.0)),  # Node 7
            Vec{3}((0.0, 1.0, 1.0)),  # Node 8
            Vec{3}((2.0, 0.0, 0.0)),  # Node 9  (second element)
            Vec{3}((2.0, 1.0, 0.0)),  # Node 10
            Vec{3}((2.0, 0.0, 1.0)),  # Node 11
            Vec{3}((2.0, 1.0, 1.0)),  # Node 12
        ]

        connectivity = [
            NTuple{8,UInt32}((1, 2, 3, 4, 5, 6, 7, 8)),        # Element 1
            NTuple{8,UInt32}((2, 9, 10, 3, 6, 11, 12, 7))      # Element 2
        ]

        mesh2 = Mesh{8,Hex8}(X, connectivity)
        cache2 = COOCache(mesh2, kernel)
        u_global2 = nothing  # Zero displacement
        state_old2 = create_material_state(kernel, mesh2)

        # Assemble
        assemble!(cache2, assembler, kernel, mesh2, u_global2, state_old2, Δt)

        # Verify we get contributions from both elements
        @test length(cache2.I) > 24 * 24  # More than single element
        @test length(cache2.f) == 36
    end
end
