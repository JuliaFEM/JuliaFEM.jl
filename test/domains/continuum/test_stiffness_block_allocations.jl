# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Focused allocation test for compute_element_stiffness_blocked!()

This test isolates just the stiffness block computation to find allocation sources.
"""

using Test
using JuliaFEM
using LinearAlgebra
using Tensors

# Import required functions
using JuliaFEM: default_integration, integration_points, compute_element_stiffness_blocked!

@testset "compute_element_stiffness_blocked! Allocation Test" begin

    # Create minimal test data
    function create_test_data()
        # Simple unit cube (8 nodes)
        X = Vec{3,Float64}[
            Vec{3}((0.0, 0.0, 0.0)),
            Vec{3}((1.0, 0.0, 0.0)),
            Vec{3}((1.0, 1.0, 0.0)),
            Vec{3}((0.0, 1.0, 0.0)),
            Vec{3}((0.0, 0.0, 1.0)),
            Vec{3}((1.0, 0.0, 1.0)),
            Vec{3}((1.0, 1.0, 1.0)),
            Vec{3}((0.0, 1.0, 1.0)),
        ]

        # Pre-allocate K_blocks matrix
        K_blocks = Matrix{Tensor{2,3,Float64,9}}(undef, 8, 8)
        fill!(K_blocks, zero(Tensor{2,3,Float64}))

        # Material
        material = LinearElastic(E=210.0e9, ν=0.3)

        # Displacement buffer (not used for LinearElastic but needed for interface)
        u_elem = zeros(Float64, 24)

        # Topology and basis
        topology = Hexahedron{8}()
        basis = Lagrange{Hexahedron{8},1}()

        # Integration points
        integration_scheme = default_integration(Hexahedron{8})
        ips = integration_points(integration_scheme, topology)

        return X, K_blocks, material, u_elem, topology, basis, ips
    end

    @testset "Warm-up and correctness" begin
        X, K_blocks, material, u_elem, topology, basis, ips = create_test_data()

        # First call (warm-up)
        compute_element_stiffness_blocked!(K_blocks, X, material, u_elem, topology, basis, ips)

        # Check result is reasonable
        @test !any(isnan, K_blocks)
        @test !any(isinf, K_blocks)
        @test any(K_blocks .!= zero(Tensor{2,3,Float64}))  # Should have non-zero values

        println("  ✓ Function produces valid output")
    end

    @testset "Zero-allocation test" begin
        X, K_blocks, material, u_elem, topology, basis, ips = create_test_data()

        # Warm-up call
        compute_element_stiffness_blocked!(K_blocks, X, material, u_elem, topology, basis, ips)

        # Reset output
        fill!(K_blocks, zero(Tensor{2,3,Float64}))

        # Measure allocations
        allocs = @allocated compute_element_stiffness_blocked!(
            K_blocks, X, material, u_elem, topology, basis, ips
        )

        println("  ✓ compute_element_stiffness_blocked!(): $(allocs) bytes allocated")

        if allocs == 0
            println("  🎉 ZERO ALLOCATIONS ACHIEVED!")
        elseif allocs < 100
            println("  ⚠ Minimal allocations (< 100 bytes)")
        else
            println("  ❌ Significant allocations detected")
            @test allocs == 0  # This will fail and show the allocation amount
        end
    end

    @testset "Binary search - Test individual components" begin
        X, K_blocks, material, u_elem, topology, basis, ips = create_test_data()

        println("\n  Testing individual operations:")

        # Test 1: Just elasticity tensor computation
        function test_elasticity_tensor(material)
            C = elasticity_tensor(material)
            return C
        end
        test_elasticity_tensor(material)  # warm-up
        allocs1 = @allocated test_elasticity_tensor(material)
        println("    [1] elasticity_tensor(): $(allocs1) bytes")

        # Test 2: Create basis vectors
        function test_basis_vectors()
            e_1 = Vec{3}((1.0, 0.0, 0.0))
            e_2 = Vec{3}((0.0, 1.0, 0.0))
            e_3 = Vec{3}((0.0, 0.0, 1.0))
            e = (e_1, e_2, e_3)
            return e
        end
        test_basis_vectors()  # warm-up
        allocs2 = @allocated test_basis_vectors()
        println("    [2] Basis vectors: $(allocs2) bytes")

        # Test 3: Loop structure with zero tensor creation
        function test_loop_structure()
            for k in 1:8, l in 1:8
                K_kl = zero(Tensor{2,3,Float64})
            end
        end
        test_loop_structure()  # warm-up
        allocs3 = @allocated test_loop_structure()
        println("    [3] Loop with zero(Tensor): $(allocs3) bytes")

        # Test 4: get_basis_derivatives call
        topology = Hexahedron{8}()
        basis = Lagrange{Hexahedron{8},1}()
        ξ_test = Vec{3}((0.0, 0.0, 0.0))
        get_basis_derivatives(topology, basis, ξ_test)  # warm-up
        allocs4 = @allocated get_basis_derivatives(topology, basis, ξ_test)
        println("    [4] get_basis_derivatives(): $(allocs4) bytes")

        # Test 5: Tensor operations (⊗, det, inv, transpose)
        function test_tensor_ops(X, dN_dξ)
            J = X[1] ⊗ dN_dξ[1]
            for i in 2:8
                J += X[i] ⊗ dN_dξ[i]
            end
            detJ = det(J)
            J_inv = inv(J)
            J_inv_T = transpose(J_inv)
            return J_inv_T
        end
        dN_dξ = get_basis_derivatives(topology, basis, ξ_test)
        test_tensor_ops(X, dN_dξ)  # warm-up
        allocs5 = @allocated test_tensor_ops(X, dN_dξ)
        println("    [5] Tensor operations (J, det, inv, transpose): $(allocs5) bytes")

        # Test 6: B-matrix computation and double contraction
        function test_b_matrix_ops(grad_k, grad_l, C, e)
            K_kl_ip = zero(Tensor{2,3,Float64})
            for α in 1:3, β in 1:3
                e_α, e_β = e[α], e[β]
                B_k_α = 0.5 * (grad_k ⊗ e_α + e_α ⊗ grad_k)
                B_l_β = 0.5 * (grad_l ⊗ e_β + e_β ⊗ grad_l)
                k_αβ = dcontract(B_k_α, dcontract(C, B_l_β))
                K_kl_ip += k_αβ * (e_α ⊗ e_β)
            end
            return K_kl_ip
        end
        C = elasticity_tensor(material)
        e = test_basis_vectors()
        J_inv_T = test_tensor_ops(X, dN_dξ)
        grad_k = J_inv_T ⋅ dN_dξ[1]
        grad_l = J_inv_T ⋅ dN_dξ[2]
        test_b_matrix_ops(grad_k, grad_l, C, e)  # warm-up
        allocs6 = @allocated test_b_matrix_ops(grad_k, grad_l, C, e)
        println("    [6] B-matrix loop with dcontract: $(allocs6) bytes")

        println()
    end

    @testset "Test full loop structure allocation" begin
        X, K_blocks, material, u_elem, topology, basis, ips = create_test_data()

        println("\n  Testing full loop structure:")

        C = elasticity_tensor(material)
        e_1, e_2, e_3 = Vec{3}((1.0, 0.0, 0.0)), Vec{3}((0.0, 1.0, 0.0)), Vec{3}((0.0, 0.0, 1.0))
        e = (e_1, e_2, e_3)

        N = 8  # Hex8

        # Warm-up
        for k in 1:N, l in 1:N
            K_kl = zero(Tensor{2,3,Float64})
            for ip in ips
                ξ = ip.ξ
                w = ip.weight
                dN_dξ = get_basis_derivatives(topology, basis, ξ)
                J = X[1] ⊗ dN_dξ[1]
                for i in 2:N
                    J += X[i] ⊗ dN_dξ[i]
                end
                detJ = det(J)
                J_inv = inv(J)
                J_inv_T = transpose(J_inv)
                grad_k = J_inv_T ⋅ dN_dξ[k]
                grad_l = J_inv_T ⋅ dN_dξ[l]
                K_kl_ip = zero(Tensor{2,3,Float64})
                for α in 1:3, β in 1:3
                    e_α, e_β = e[α], e[β]
                    B_k_α = 0.5 * (grad_k ⊗ e_α + e_α ⊗ grad_k)
                    B_l_β = 0.5 * (grad_l ⊗ e_β + e_β ⊗ grad_l)
                    k_αβ = dcontract(B_k_α, dcontract(C, B_l_β))
                    K_kl_ip += k_αβ * (e_α ⊗ e_β)
                end
                K_kl += K_kl_ip * detJ * w
            end
            K_blocks[k, l] = K_kl
        end

        # Measure allocations
        fill!(K_blocks, zero(Tensor{2,3,Float64}))
        allocs = @allocated begin
            for k in 1:N, l in 1:N
                K_kl = zero(Tensor{2,3,Float64})
                for ip in ips
                    ξ = ip.ξ
                    w = ip.weight
                    dN_dξ = get_basis_derivatives(topology, basis, ξ)
                    J = X[1] ⊗ dN_dξ[1]
                    for i in 2:N
                        J += X[i] ⊗ dN_dξ[i]
                    end
                    detJ = det(J)
                    J_inv = inv(J)
                    J_inv_T = transpose(J_inv)
                    grad_k = J_inv_T ⋅ dN_dξ[k]
                    grad_l = J_inv_T ⋅ dN_dξ[l]
                    K_kl_ip = zero(Tensor{2,3,Float64})
                    for α in 1:3, β in 1:3
                        e_α, e_β = e[α], e[β]
                        B_k_α = 0.5 * (grad_k ⊗ e_α + e_α ⊗ grad_k)
                        B_l_β = 0.5 * (grad_l ⊗ e_β + e_β ⊗ grad_l)
                        k_αβ = dcontract(B_k_α, dcontract(C, B_l_β))
                        K_kl_ip += k_αβ * (e_α ⊗ e_β)
                    end
                    K_kl += K_kl_ip * detJ * w
                end
                K_blocks[k, l] = K_kl
            end
        end

        println("    Full loop (inline): $(allocs) bytes allocated")
        println("    Iterations: $(N*N) node pairs × $(length(ips)) ips = $(N*N*length(ips)) total")
        println("    Bytes per iteration: $(allocs / (N*N*length(ips)))")
    end

end

println("\n" * "="^70)
println("STIFFNESS BLOCK ALLOCATION TEST SUMMARY")
println("="^70)
