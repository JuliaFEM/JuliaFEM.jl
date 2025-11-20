# Test compute_block! function (Phase 3)

@testset "compute_block!" begin
    kernel = create_test_kernel()
    mesh = create_test_mesh()

    # Create caches
    N = 8      # Nodes per element (Hex8)
    NIP = 8    # Integration points (Gauss{2} for Hex8)

    geometry_cache = JuliaFEM.create_geometry_cache(N, NIP)
    element_cache = JuliaFEM.create_element_cache(mesh, kernel)
    material_cache = JuliaFEM.create_material_cache(kernel.material, NIP)

    # Prepare caches (Phases 1a, 1b, 2)
    elem_id = 1
    u_global = nothing  # Zero displacement
    state_old = create_material_state(kernel, mesh)
    Δt = 0.01

    JuliaFEM.update_geometry_cache!(geometry_cache, element_cache, kernel, elem_id, mesh)
    JuliaFEM.update_element_cache!(element_cache, kernel, elem_id, mesh, u_global)
    JuliaFEM.update_material_cache!(material_cache, geometry_cache, kernel.material,
        element_cache, state_old, elem_id, Δt)

    @testset "Correctness" begin
        # Pre-allocate K_blocks matrix
        K_blocks = Matrix{Tensor{2,3,Float64,9}}(undef, N, N)

        # Compute a single stiffness block K[1,1]
        JuliaFEM.compute_block!(K_blocks, geometry_cache, material_cache, 1, 1)
        K_11 = K_blocks[1, 1]

        # Verify output type
        @test K_11 isa Tensor{2,3,Float64}

        # Verify symmetry (for linear elastic)
        @test K_11 ≈ transpose(K_11) rtol = 1e-14  # Relative tolerance for large values

        # Verify positive diagonal (stiffness)
        for α in 1:3
            @test K_11[α, α] > 0.0
        end

        # Compute off-diagonal block K[1,2]
        JuliaFEM.compute_block!(K_blocks, geometry_cache, material_cache, 1, 2)
        K_12 = K_blocks[1, 2]
        @test K_12 isa Tensor{2,3,Float64}
    end

    @testset "Zero Allocations" begin
        # Pre-allocate K_blocks matrix
        K_blocks = Matrix{Tensor{2,3,Float64,9}}(undef, N, N)

        # Warm-up call
        JuliaFEM.compute_block!(K_blocks, geometry_cache, material_cache, 1, 1)

        # Test zero allocations for single call
        allocs = @allocated JuliaFEM.compute_block!(K_blocks, geometry_cache, material_cache, 1, 1)
        @test allocs == 0

        # Test allocations for loop - must be EXACTLY zero
        for k in 1:N, l in 1:N
            JuliaFEM.compute_block!(K_blocks, geometry_cache, material_cache, k, l)
        end

        allocs_loop = @allocated begin
            for k in 1:N, l in 1:N
                JuliaFEM.compute_block!(K_blocks, geometry_cache, material_cache, k, l)
            end
        end
        @test allocs_loop == 0
    end
end
