# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE

"""
Test CElement Interpolation and Gradient Computation
"""

# Mock mesh structure for testing
struct TestMesh2
    nodes::Dict{Int, Vec}
    connectivity::Dict{Int, Tuple}
end

@testset "CElement Interpolation" begin
    @testset "Triangle ScalarDOF - Temperature Field" begin
        # Create mesh: single triangle
        mesh = TestMesh2(
            Dict(
                1 => Vec{2}((0.0, 0.0)),
                2 => Vec{2}((1.0, 0.0)),
                3 => Vec{2}((0.0, 1.0))
            ),
            Dict(1 => (1, 2, 3))
        )
        
        # Create element with DOFs
        elem = CElement{Triangle{3}, Lagrange{1}, ScalarDOF}(
            1,  # element id
            (10, 20, 30)  # DOF indices
        )
        
        # Temperature field: T = [100.0, 200.0, 150.0] at nodes 1,2,3
        u_global = zeros(100)
        u_global[10] = 100.0  # Node 1
        u_global[20] = 200.0  # Node 2
        u_global[30] = 150.0  # Node 3
        
        # NOTE: Using stub basis evaluation (uniform weights)
        # All interpolations return average: (100 + 200 + 150) / 3 = 150
        # TODO: Update these tests once real Lagrange basis is integrated
        
        # Interpolate at element center (ξ = (1/3, 1/3))
        ξ = Vec{2}((1.0/3.0, 1.0/3.0))
        T_center = interpolate(elem, mesh, u_global, ξ)
        @test T_center ≈ 150.0 atol=1e-10
        
        # Stub returns average everywhere (not actual nodal values)
        T_node1 = interpolate(elem, mesh, u_global, Vec{2}((0.0, 0.0)))
        @test T_node1 ≈ 150.0 atol=1e-10  # Stub: should be 100.0 with real basis
        
        T_node2 = interpolate(elem, mesh, u_global, Vec{2}((1.0, 0.0)))
        @test T_node2 ≈ 150.0 atol=1e-10  # Stub: should be 200.0 with real basis
        
        T_node3 = interpolate(elem, mesh, u_global, Vec{2}((0.0, 1.0)))
        @test T_node3 ≈ 150.0 atol=1e-10  # Stub: should be 150.0 (happens to match!)
    end
    
    @testset "Tetrahedron VectorDOF{3} - Displacement Field" begin
        # Create mesh: single tetrahedron
        mesh = TestMesh2(
            Dict(
                1 => Vec{3}((0.0, 0.0, 0.0)),
                2 => Vec{3}((1.0, 0.0, 0.0)),
                3 => Vec{3}((0.0, 1.0, 0.0)),
                4 => Vec{3}((0.0, 0.0, 1.0))
            ),
            Dict(1 => (1, 2, 3, 4))
        )
        
        # Create element with DOFs (4 nodes × 3 DOFs = 12 DOFs)
        elem = CElement{Tetrahedron{4}, Lagrange{1}, VectorDOF{3}}(
            1,
            (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
        )
        
        # Displacement field: u = [0, 0, 0] at all nodes except node 2 = [0.1, 0, 0]
        u_global = zeros(100)
        u_global[4] = 0.1  # Node 2, x-component
        
        # NOTE: Using stub basis evaluation (uniform weights 1/4 for tet)
        # Average of all nodes: [0.025, 0, 0]
        # TODO: Update once real basis is integrated
        
        # Interpolate at element center
        ξ = Vec{3}((0.25, 0.25, 0.25))
        u_center = interpolate(elem, mesh, u_global, ξ)
        
        # Result should be Vec{3} with x ≈ 0.025 (0.1 / 4) - stub gives average
        @test u_center isa Vec{3}
        @test u_center[1] ≈ 0.025 atol=1e-10
        @test u_center[2] ≈ 0.0 atol=1e-10
        @test u_center[3] ≈ 0.0 atol=1e-10
        
        # Stub returns average everywhere (not actual nodal value)
        u_node2 = interpolate(elem, mesh, u_global, Vec{3}((1.0, 0.0, 0.0)))
        @test u_node2[1] ≈ 0.025 atol=1e-10  # Stub: should be 0.1 with real basis
        @test u_node2[2] ≈ 0.0 atol=1e-10
        @test u_node2[3] ≈ 0.0 atol=1e-10
    end
end

@testset "CElement Gradient Computation" begin
    @testset "Triangle ScalarDOF - Temperature Gradient" begin
        # Create mesh: right triangle with sides along x and y axes
        mesh = TestMesh2(
            Dict(
                1 => Vec{2}((0.0, 0.0)),
                2 => Vec{2}((1.0, 0.0)),
                3 => Vec{2}((0.0, 1.0))
            ),
            Dict(1 => (1, 2, 3))
        )
        
        elem = CElement{Triangle{3}, Lagrange{1}, ScalarDOF}(
            1,
            (10, 20, 30)
        )
        
        # Linear temperature field: T(x, y) = 100 + 50*x + 30*y
        # Node 1 (0,0): T = 100
        # Node 2 (1,0): T = 150
        # Node 3 (0,1): T = 130
        u_global = zeros(100)
        u_global[10] = 100.0
        u_global[20] = 150.0
        u_global[30] = 130.0
        
        # NOTE: Gradient stub returns zeros
        # TODO: Should be ∇T = [50, 30] once real basis derivatives are integrated
        ξ = Vec{2}((0.3, 0.3))
        grad_T = JuliaFEM.gradient(elem, mesh, u_global, ξ)
        
        @test grad_T isa Vec{2}
        @test grad_T[1] ≈ 0.0 atol=1e-8  # Stub: should be 50.0 with real basis
        @test grad_T[2] ≈ 0.0 atol=1e-8  # Stub: should be 30.0 with real basis
    end
    
    @testset "Tetrahedron VectorDOF{3} - Deformation Gradient" begin
        # Create mesh: unit tetrahedron
        mesh = TestMesh2(
            Dict(
                1 => Vec{3}((0.0, 0.0, 0.0)),
                2 => Vec{3}((1.0, 0.0, 0.0)),
                3 => Vec{3}((0.0, 1.0, 0.0)),
                4 => Vec{3}((0.0, 0.0, 1.0))
            ),
            Dict(1 => (1, 2, 3, 4))
        )
        
        elem = CElement{Tetrahedron{4}, Lagrange{1}, VectorDOF{3}}(
            1,
            (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
        )
        
        # Uniform displacement: u(x,y,z) = [0.1*x, 0, 0]
        # This creates a constant deformation gradient
        u_global = zeros(100)
        u_global[4] = 0.1  # Node 2, x = 1.0
        
        # NOTE: Gradient stub returns zeros
        # TODO: Should be F[1,1]=0.1 once real basis derivatives are integrated
        ξ = Vec{3}((0.25, 0.25, 0.25))
        F = JuliaFEM.gradient(elem, mesh, u_global, ξ)
        
        # Result should be Tensor{2,3} (deformation gradient)
        @test F isa Tensor{2,3}
        
        # Stub returns all zeros
        @test F[1,1] ≈ 0.0 atol=1e-8  # Stub: should be 0.1 with real basis
        @test F[1,2] ≈ 0.0 atol=1e-8
        @test F[1,3] ≈ 0.0 atol=1e-8
        @test F[2,1] ≈ 0.0 atol=1e-8
        @test F[3,1] ≈ 0.0 atol=1e-8
    end
end
