# Test compute_block! function (Phase 3)

using JuliaFEM
using Tensors

@testset "compute_block!" begin
    # Setup arrays directly without caches to eliminate any cache-related allocations
    N = 8      # Nodes per element (Hex8)
    NIP = 8    # Integration points (Gauss{2} for Hex8)

    # Create shape function gradient matrix directly [NIP × N]
    # Typical gradient values for Hex8 element
    ∇N_data = Matrix{Vec{3,Float64}}(undef, NIP, N)
    for q in 1:NIP, k in 1:N
        # Realistic gradient values
        ∇N_data[q, k] = Vec{3}((0.1 * k + 0.05 * q, 0.15 * k - 0.03 * q, 0.12 * k + 0.02 * q))
    end

    # Jacobian determinant times weight at each integration point
    detJ_w = fill(0.125, NIP)  # Typical value for unit cube

    # Material tangent modulus (elasticity tensor) at each integration point
    # LinearElastic: E=210e9, ν=0.3
    material = JuliaFEM.LinearElastic(E=210e9, ν=0.3)
    D_single = JuliaFEM.elasticity_tensor(material)
    D_array = fill(D_single, NIP)

    @testset "Correctness" begin
        # Pre-allocate K_blocks matrix
        K_blocks = Matrix{Tensor{2,3,Float64,9}}(undef, N, N)

        # Compute a single stiffness block K[1,1]
        JuliaFEM.compute_block!(K_blocks, ∇N_data, detJ_w, D_array, 1, 1)
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
        JuliaFEM.compute_block!(K_blocks, ∇N_data, detJ_w, D_array, 1, 2)
        K_12 = K_blocks[1, 2]
        @test K_12 isa Tensor{2,3,Float64}
    end

    @testset "Zero Allocations" begin
        # Pre-allocate K_blocks matrix
        K_blocks = Matrix{Tensor{2,3,Float64,9}}(undef, N, N)

        # Warm-up call
        JuliaFEM.compute_block!(K_blocks, ∇N_data, detJ_w, D_array, 1, 1)

        # Test zero allocations for single call - THE ACTUAL GUARANTEE
        allocs = @allocated JuliaFEM.compute_block!(K_blocks, ∇N_data, detJ_w, D_array, 1, 1)
        @test allocs == 0  # CRITICAL: compute_block! has zero allocations!
    end
end
