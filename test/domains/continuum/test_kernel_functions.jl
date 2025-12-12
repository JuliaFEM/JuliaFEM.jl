# Test kernel.jl functions (compute_block_at_point)

@testset "Kernel Functions (kernel.jl)" begin
    @testset "compute_block_at_point" begin
        # Material properties (Steel)
        E = 210e9  # Pa
        ν = 0.3
        material = LinearElastic(E=E, ν=ν)

        # Get elasticity tensor
        C = elasticity_tensor(material)

        # Sample gradients (arbitrary but realistic)
        grad_k = Vec{3}((0.1, 0.2, 0.3))
        grad_l = Vec{3}((0.4, 0.5, 0.6))

        @testset "Correctness" begin
            # Compute stiffness block at point
            K_kl = JuliaFEM.compute_block_at_point(grad_k, grad_l, C)

            # Verify output type
            @test K_kl isa Tensor{2,3,Float64}

            # Verify all components are finite
            @test all(isfinite, K_kl)

            # Verify symmetry for identical gradients
            K_same = JuliaFEM.compute_block_at_point(grad_k, grad_k, C)
            @test K_same ≈ transpose(K_same) rtol=1e-14  # Relative tolerance for large values
        end

        @testset "Zero Allocations" begin
            # Warm-up call
            JuliaFEM.compute_block_at_point(grad_k, grad_l, C)

            # Test zero allocations
            allocs = @allocated JuliaFEM.compute_block_at_point(grad_k, grad_l, C)
            @test allocs == 0
        end

        @testset "Consistency with compute_block!" begin
            # Create a simple test case where we can compare
            # compute_block_at_point (single IP) with compute_block! (integrated)
            kernel = create_test_kernel()
            mesh = create_test_mesh()

            N = 8
            NIP = 8

            geometry_cache = JuliaFEM.create_geometry_cache(N, NIP)
            element_cache = JuliaFEM.create_element_cache(mesh, kernel)
            material_cache = JuliaFEM.create_material_cache(kernel.material, NIP)

            # Update caches
            elem_id = 1
            JuliaFEM.update_geometry_cache!(geometry_cache, element_cache, kernel, elem_id, mesh)
            JuliaFEM.update_element_cache!(element_cache, kernel, elem_id, mesh, nothing)
            JuliaFEM.update_material_cache!(material_cache, geometry_cache, kernel.material,
                element_cache, nothing, elem_id, 0.0)

            # Manual integration using compute_block_at_point
            K_manual = zero(Tensor{2,3,Float64})
            for q in 1:NIP
                𝔻 = JuliaFEM.get_tangent(material_cache, q)
                grad_k_test = geometry_cache.∇N_data[q, 1]
                grad_l_test = geometry_cache.∇N_data[q, 2]
                detJ_w = geometry_cache.detJ_w[q]

                K_ip = JuliaFEM.compute_block_at_point(grad_k_test, grad_l_test, 𝔻)
                K_manual += K_ip * detJ_w
            end

            # Automatic integration using compute_block!
            K_blocks = Matrix{Tensor{2,3,Float64,9}}(undef, N, N)
            JuliaFEM.compute_block!(
                K_blocks,
                geometry_cache.∇N_data,
                geometry_cache.detJ_w,
                [JuliaFEM.get_tangent(material_cache, q) for q in 1:length(material_cache.states)],
                1, 2
            )
            K_auto = K_blocks[1, 2]

            # Should match (within numerical precision)
            @test K_manual ≈ K_auto rtol = 1e-12
        end
    end
end
