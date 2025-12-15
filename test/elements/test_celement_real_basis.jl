using Test
using JuliaFEM
using JuliaFEM: CElement, ScalarDOF, VectorDOF, interpolate, gradient
using Tensors

"""
Test suite for CElement with REAL basis functions (not stubs).

This test file verifies that CElement correctly uses the actual Lagrange
basis functions from basis_generated.jl, not the stub implementations.
"""

@testset "CElement Real Basis Functions" begin
    
    @testset "Triangle Tri3 Linear Interpolation" begin
        # Create a linear triangle element
        # Nodes at vertices: (0,0), (1,0), (0,1)
        nodes = [
            Vec{2}((0.0, 0.0)),  # Node 1
            Vec{2}((1.0, 0.0)),  # Node 2
            Vec{2}((0.0, 1.0))   # Node 3
        ]
        
        # Simple mesh structure
        mesh = (
            connectivity = [[1, 2, 3]],
            nodes = nodes
        )
        
        # Create element with scalar DOF
        elem = CElement{Tri3, Lagrange{1}, ScalarDOF}(1, (1, 2, 3))
        
        # Test nodal interpolation: At nodes, basis functions should be 1 at that node, 0 elsewhere
        # At node 1: ξ = (0, 0) → should get u[1] exactly
        u_global = [10.0, 20.0, 30.0]  # Values at nodes 1, 2, 3
        
        # At parametric coord (0, 0) = node 1
        val1 = interpolate(elem, mesh, u_global, Vec{2}((0.0, 0.0)))
        @test val1 ≈ 10.0 atol=1e-12
        
        # At parametric coord (1, 0) = node 2
        val2 = interpolate(elem, mesh, u_global, Vec{2}((1.0, 0.0)))
        @test val2 ≈ 20.0 atol=1e-12
        
        # At parametric coord (0, 1) = node 3
        val3 = interpolate(elem, mesh, u_global, Vec{2}((0.0, 1.0)))
        @test val3 ≈ 30.0 atol=1e-12
        
        # Test centroid: ξ = (1/3, 1/3) should give average
        centroid = interpolate(elem, mesh, u_global, Vec{2}(1.0/3.0, 1.0/3.0))
        expected_centroid = (10.0 + 20.0 + 30.0) / 3.0
        @test centroid ≈ expected_centroid atol=1e-12
        
        println("✓ Tri3 linear interpolation: Exact at nodes, correct at centroid")
    end
    
    @testset "Triangle Tri3 Linear Gradient" begin
        # Same triangle as above
        nodes = [
            Vec{2}((0.0, 0.0)),
            Vec{2}((1.0, 0.0)),
            Vec{2}((0.0, 1.0))
        ]
        
        mesh = (
            connectivity = [[1, 2, 3]],
            nodes = nodes
        )
        
        elem = CElement{Tri3, Lagrange{1}, ScalarDOF}(1, (1, 2, 3))
        
        # Linear field: u = x + 2y (gradient should be [1, 2])
        # At node 1 (0,0): u = 0
        # At node 2 (1,0): u = 1
        # At node 3 (0,1): u = 2
        u_global = [0.0, 1.0, 2.0]
        
        # Gradient should be constant [1, 2] everywhere for linear element
        grad_center = gradient(elem, mesh, u_global, Vec{2}(1.0/3.0, 1.0/3.0))
        @test grad_center[1] ≈ 1.0 atol=1e-10
        @test grad_center[2] ≈ 2.0 atol=1e-10
        
        # Gradient should be same at different points
        grad_node1 = gradient(elem, mesh, u_global, Vec{2}((0.0, 0.0)))
        @test grad_node1[1] ≈ 1.0 atol=1e-10
        @test grad_node1[2] ≈ 2.0 atol=1e-10
        
        println("✓ Tri3 gradient: Constant for linear field")
    end
    
    @testset "Tetrahedron Tet4 Linear Interpolation" begin
        # Create a linear tetrahedron
        # Standard reference tet: (0,0,0), (1,0,0), (0,1,0), (0,0,1)
        nodes = [
            Vec{3}((0.0, 0.0, 0.0)),  # Node 1
            Vec{3}((1.0, 0.0, 0.0)),  # Node 2
            Vec{3}((0.0, 1.0, 0.0)),  # Node 3
            Vec{3}((0.0, 0.0, 1.0))   # Node 4
        ]
        
        mesh = (
            connectivity = [[1, 2, 3, 4]],
            nodes = nodes
        )
        
        elem = CElement{Tet4, Lagrange{1}, ScalarDOF}(1, (1, 2, 3, 4))
        
        # Test nodal interpolation
        u_global = [5.0, 15.0, 25.0, 35.0]
        
        # At node 1: ξ = (0, 0, 0)
        val1 = interpolate(elem, mesh, u_global, Vec{3}((0.0, 0.0, 0.0)))
        @test val1 ≈ 5.0 atol=1e-12
        
        # At node 2: ξ = (1, 0, 0)
        val2 = interpolate(elem, mesh, u_global, Vec{3}((1.0, 0.0, 0.0)))
        @test val2 ≈ 15.0 atol=1e-12
        
        # At node 3: ξ = (0, 1, 0)
        val3 = interpolate(elem, mesh, u_global, Vec{3}((0.0, 1.0, 0.0)))
        @test val3 ≈ 25.0 atol=1e-12
        
        # At node 4: ξ = (0, 0, 1)
        val4 = interpolate(elem, mesh, u_global, Vec{3}((0.0, 0.0, 1.0)))
        @test val4 ≈ 35.0 atol=1e-12
        
        # Test centroid: ξ = (1/4, 1/4, 1/4)
        centroid = interpolate(elem, mesh, u_global, Vec{3}((0.25, 0.25, 0.25)))
        expected = (5.0 + 15.0 + 25.0 + 35.0) / 4.0
        @test centroid ≈ expected atol=1e-12
        
        println("✓ Tet4 linear interpolation: Exact at nodes, correct at centroid")
    end
    
    @testset "Tetrahedron Tet4 Linear Gradient" begin
        # Same tet as above
        nodes = [
            Vec{3}((0.0, 0.0, 0.0)),
            Vec{3}((1.0, 0.0, 0.0)),
            Vec{3}((0.0, 1.0, 0.0)),
            Vec{3}((0.0, 0.0, 1.0))
        ]
        
        mesh = (
            connectivity = [[1, 2, 3, 4]],
            nodes = nodes
        )
        
        elem = CElement{Tet4, Lagrange{1}, ScalarDOF}(1, (1, 2, 3, 4))
        
        # Linear field: u = 2x + 3y + 4z → gradient = [2, 3, 4]
        # At nodes: u = [0, 2, 3, 4]
        u_global = [0.0, 2.0, 3.0, 4.0]
        
        # Gradient should be constant [2, 3, 4] everywhere
        grad_center = gradient(elem, mesh, u_global, Vec{3}((0.25, 0.25, 0.25)))
        @test grad_center[1] ≈ 2.0 atol=1e-10
        @test grad_center[2] ≈ 3.0 atol=1e-10
        @test grad_center[3] ≈ 4.0 atol=1e-10
        
        println("✓ Tet4 gradient: Constant for linear field")
    end
    
    @testset "Vector DOF Deformation Gradient" begin
        # Test 2D vector DOF (displacement field)
        nodes = [
            Vec{2}((0.0, 0.0)),
            Vec{2}((1.0, 0.0)),
            Vec{2}((0.0, 1.0))
        ]
        
        mesh = (
            connectivity = [[1, 2, 3]],
            nodes = nodes
        )
        
        # Element with 2D vector DOF
        elem = CElement{Tri3, Lagrange{1}, VectorDOF{2}}(1, (1, 2, 3, 4, 5, 6))
        
        # Displacement field: u = [x, 2y] → F = [∂u₁/∂x ∂u₁/∂y; ∂u₂/∂x ∂u₂/∂y] = [1 0; 0 2]
        # At nodes: u = [[0,0], [1,0], [0,2]]
        u_global = [0.0, 0.0,  # Node 1: (ux, uy)
                    1.0, 0.0,  # Node 2
                    0.0, 2.0]  # Node 3
        
        # Deformation gradient
        F = gradient(elem, mesh, u_global, Vec{2}(1.0/3.0, 1.0/3.0))
        
        # Should be [1 0; 0 2] for linear displacement
        @test F[1,1] ≈ 1.0 atol=1e-10  # ∂u₁/∂x
        @test F[1,2] ≈ 0.0 atol=1e-10  # ∂u₁/∂y
        @test F[2,1] ≈ 0.0 atol=1e-10  # ∂u₂/∂x
        @test F[2,2] ≈ 2.0 atol=1e-10  # ∂u₂/∂y
        
        println("✓ VectorDOF deformation gradient: Correct for linear displacement")
    end
    
    @testset "Partition of Unity (Completeness)" begin
        # Basis functions should sum to 1 at any point
        nodes = [
            Vec{2}((0.0, 0.0)),
            Vec{2}((1.0, 0.0)),
            Vec{2}((0.0, 1.0))
        ]
        
        mesh = (
            connectivity = [[1, 2, 3]],
            nodes = nodes
        )
        
        elem = CElement{Tri3, Lagrange{1}, ScalarDOF}(1, (1, 2, 3))
        
        # Test at several points
        test_points = [
            Vec{2}((0.0, 0.0)),
            Vec{2}((0.5, 0.0)),
            Vec{2}((0.0, 0.5)),
            Vec{2}((0.5, 0.5)),
            Vec{2}((0.33, 0.33)),
            Vec{2}((0.1, 0.7))
        ]
        
        u_ones = [1.0, 1.0, 1.0]  # If all nodal values = 1, result should be 1
        
        for ξ in test_points
            # Skip if outside element (u + v > 1)
            if ξ[1] + ξ[2] > 1.0
                continue
            end
            
            val = interpolate(elem, mesh, u_ones, ξ)
            @test val ≈ 1.0 atol=1e-12
        end
        
        println("✓ Partition of unity: Sum of basis = 1 at all points")
    end
    
    @testset "Linear Reproduction (Consistency)" begin
        # Linear functions should be reproduced exactly
        nodes = [
            Vec{2}((0.0, 0.0)),
            Vec{2}((2.0, 0.0)),
            Vec{2}((0.0, 3.0))
        ]
        
        mesh = (
            connectivity = [[1, 2, 3]],
            nodes = nodes
        )
        
        elem = CElement{Tri3, Lagrange{1}, ScalarDOF}(1, (1, 2, 3))
        
        # Linear function: u(x,y) = 5 + 2x + 3y
        u_nodal = [
            5.0,              # At (0,0): 5
            5.0 + 2.0*2.0,   # At (2,0): 9
            5.0 + 3.0*3.0    # At (0,3): 14
        ]
        
        # Test at arbitrary physical points
        test_points = [
            (Vec{2}((0.5, 0.0)), Vec{2}((1.0, 0.0))),   # (ξ, physical)
            (Vec{2}((0.0, 0.5)), Vec{2}((0.0, 1.5))),
            (Vec{2}((0.25, 0.25)), Vec{2}((0.5, 0.75)))
        ]
        
        for (ξ, x_phys) in test_points
            val = interpolate(elem, mesh, u_nodal, ξ)
            expected = 5.0 + 2.0*x_phys[1] + 3.0*x_phys[2]
            @test val ≈ expected atol=1e-10
        end
        
        println("✓ Linear reproduction: Linear functions reproduced exactly")
    end
    
end  # @testset "CElement Real Basis Functions"

println("\n" * "="^70)
println("CElement Real Basis Test Summary")
println("="^70)
println("✅ All tests verify that CElement uses REAL Lagrange basis functions")
println("✅ Interpolation: Exact at nodes, correct partition of unity")
println("✅ Gradient: Constant for linear elements, correct deformation gradient")
println("✅ Math properties: Completeness and consistency verified")
println("="^70)
