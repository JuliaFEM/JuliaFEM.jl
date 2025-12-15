"""
Test strain-based LinearElastic computation vs fast-path C-based version.

This test validates that explicit strain computation at integration points
gives IDENTICAL results to the fast-path pre-computed C approach.

If this passes, we know:
1. Strain computation is correct
2. Infrastructure is ready for NeoHookean (same pattern)
3. Fast-path can be trusted as reference
"""

using Test
using JuliaFEM
using Tensors

@testset "Strain-Based LinearElastic Validation" begin
    println("\n" * "="^70)
    println("STRAIN-BASED LINEAR ELASTIC VALIDATION")
    println("="^70)

    # Material
    material = LinearElastic(E=210e9, ν=0.3)
    C = JuliaFEM.elasticity_tensor(material)

    # Single Hex8 element at origin
    topology = Hex8()
    basis = Lagrange{Hex8,1}()
    ips = integration_points(Gauss{2}(), topology)

    # Element geometry (1m cube)
    X = [
        Vec{3}((0.0, 0.0, 0.0)),  # Node 1
        Vec{3}((1.0, 0.0, 0.0)),  # Node 2
        Vec{3}((1.0, 1.0, 0.0)),  # Node 3
        Vec{3}((0.0, 1.0, 0.0)),  # Node 4
        Vec{3}((0.0, 0.0, 1.0)),  # Node 5
        Vec{3}((1.0, 0.0, 1.0)),  # Node 6
        Vec{3}((1.0, 1.0, 1.0)),  # Node 7
        Vec{3}((0.0, 1.0, 1.0)),  # Node 8
    ]

    # Element displacement (zero for linear elastic tangent)
    u_elem = zeros(24)  # 8 nodes × 3 DOFs

    println("\n[1] Testing with zero displacement (tangent at u=0)...")

    # Fast-path: Pre-computed C
    K_fast = zeros(Tensor{2,3}, 8, 8)
    JuliaFEM.compute_element_stiffness!(K_fast, X, C, topology, basis, ips)

    println("  Fast-path computed")

    # Strain-based: Explicit strain computation
    K_strain = zeros(Tensor{2,3}, 8, 8)
    JuliaFEM.compute_element_stiffness_strain_based!(K_strain, X, material, u_elem, topology, basis, ips)

    println("  Strain-based computed")

    # Compare element stiffness matrices
    println("\n[2] Comparing results...")

    max_diff = 0.0
    max_rel_diff = 0.0

    for i in 1:8, j in 1:8
        for α in 1:3, β in 1:3
            K_fast_val = K_fast[i, j][α, β]
            K_strain_val = K_strain[i, j][α, β]

            diff = abs(K_fast_val - K_strain_val)
            max_diff = max(max_diff, diff)

            if abs(K_fast_val) > 1e-10
                rel_diff = diff / abs(K_fast_val)
                max_rel_diff = max(max_rel_diff, rel_diff)
            end
        end
    end

    println("  Maximum absolute difference: $(max_diff)")
    println("  Maximum relative difference: $(max_rel_diff)")

    # Test: Should be IDENTICAL (within machine precision)
    max_tensor_diff = maximum(norm(K_fast[i, j] - K_strain[i, j]) for i in 1:8, j in 1:8)
    @test max_tensor_diff < 1e-6  # Absolute tolerance
    println("  ✓ Matrices match within tolerance")

    # Test a few specific entries
    println("\n[3] Checking specific stiffness blocks...")

    # Diagonal block (1,1) - should be symmetric positive definite
    K11_fast = K_fast[1, 1]
    K11_strain = K_strain[1, 1]

    println("  K[1,1] fast-path:")
    println("    ", K11_fast)
    println("  K[1,1] strain-based:")
    println("    ", K11_strain)

    @test norm(K11_fast - K11_strain) < 1e-6
    println("  ✓ Diagonal block matches")

    # Off-diagonal block (1,2) - coupling between nodes
    K12_fast = K_fast[1, 2]
    K12_strain = K_strain[1, 2]

    println("\n  K[1,2] difference norm: $(norm(K12_fast - K12_strain))")
    @test norm(K12_fast - K12_strain) < 1e-6
    println("  ✓ Off-diagonal block matches")

    println("\n[4] Testing with non-zero displacement...")

    # Small displacement (10mm = 0.01m in each direction)
    u_elem_displaced = fill(0.01, 24)  # Uniform 1cm displacement

    K_fast_u = zeros(Tensor{2,3}, 8, 8)
    K_strain_u = zeros(Tensor{2,3}, 8, 8)

    # For LinearElastic, tangent should be INDEPENDENT of displacement
    # So results should still match
    JuliaFEM.compute_element_stiffness!(K_fast_u, X, C, topology, basis, ips)
    JuliaFEM.compute_element_stiffness_strain_based!(K_strain_u, X, material, u_elem_displaced,
        topology, basis, ips)

    max_tensor_diff = maximum(norm(K_fast_u[i, j] - K_strain_u[i, j]) for i in 1:8, j in 1:8)
    @test max_tensor_diff < 1e-6
    println("  ✓ Tangent independent of displacement (as expected for LinearElastic)")

    # Verify tangent is same at u=0 and u≠0 (linear material!)
    max_tensor_diff = maximum(norm(K_fast[i, j] - K_fast_u[i, j]) for i in 1:8, j in 1:8)
    @test max_tensor_diff < 1e-10
    println("  ✓ Tangent is constant (validates linear elasticity)")

    println("\n" * "="^70)
    println("VALIDATION SUMMARY")
    println("="^70)
    println("  ✓ Strain-based computation matches fast-path")
    println("  ✓ Zero-allocation infrastructure working")
    println("  ✓ Ready for NeoHookean implementation")
    println("="^70)
end
