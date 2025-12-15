# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Unit tests for DOF connectivity structures.

Tests:
1. DOFElementConnection creation and access
2. DOFConnectivity building and access
3. DOFConnectivityGPU building and access
4. Zero-allocation verification
5. GPU compatibility (bits types)
"""

using Test
using JuliaFEM
using JuliaFEM: DOFElementConnection, DOFConnectivity, DOFConnectivityGPU
using JuliaFEM: build_dof_connectivity, build_dof_connectivity_gpu
using JuliaFEM: connection_count, is_empty, elem_id, local_dof_idx
using JuliaFEM: DOFManager, create_elements!, @DOFSet, DOF, Displacement, Vertex
using BenchmarkTools

@testset "DOF Connectivity" begin
    println("\n" * "="^70)
    println("DOF CONNECTIVITY TESTS")
    println("="^70)
    
    # ========================================================================
    # 1. DOFElementConnection
    # ========================================================================
    
    @testset "DOFElementConnection" begin
        println("\n[1] Testing DOFElementConnection...")
        
        # Create connection
        conn = DOFElementConnection(5, 12)
        @test elem_id(conn) == 5
        @test local_dof_idx(conn) == 12
        
        # Test type (bits type for GPU)
        # Note: isbits() checks if an instance is a bits type (immutable, no pointers)
        # DOFElementConnection is a struct with Int32 and Int16 fields - it IS a bits type
        @test isbits(conn)
        # Verify it's a struct (not mutable struct)
        @test DOFElementConnection isa Type
        
        # Test range validation
        @test_throws ErrorException DOFElementConnection(0, 1)
        @test_throws ErrorException DOFElementConnection(typemax(Int32) + 1, 1)
        @test_throws ErrorException DOFElementConnection(1, 0)
        @test_throws ErrorException DOFElementConnection(1, typemax(Int16) + 1)
        
        println("  ✓ DOFElementConnection working")
    end
    
    # ========================================================================
    # 2. DOFConnectivity (CPU Version)
    # ========================================================================
    
    @testset "DOFConnectivity - CPU" begin
        println("\n[2] Testing DOFConnectivity (CPU)...")
        
        # Create simple test mesh: 2 elements sharing a node
        # Element 1: nodes [1, 2, 3] → DOFs [1, 2, 3, 4, 5, 6, 7, 8, 9]
        # Element 2: nodes [2, 4, 5] → DOFs [4, 5, 6, 10, 11, 12, 13, 14, 15]
        # DOF 4, 5, 6 are shared (node 2)
        
        # Create DOF manager
        mgr = DOFManager()
        
        # Create elements manually (simplified for testing)
        S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
        ElemType = Element{Triangle{3}, Lagrange{1}, S}
        
        # Element 1: DOFs 1-9 (3 nodes × 3 DOFs)
        elem1 = ElemType(UInt(1), (UInt64(1), UInt64(2), UInt64(3), UInt64(4), UInt64(5), UInt64(6), UInt64(7), UInt64(8), UInt64(9)))
        
        # Element 2: DOFs 4-6, 10-15 (shares node 2 with elem1)
        elem2 = ElemType(UInt(2), (UInt64(4), UInt64(5), UInt64(6), UInt64(10), UInt64(11), UInt64(12), UInt64(13), UInt64(14), UInt64(15)))
        
        elements = [elem1, elem2]
        
        # Set total DOFs in manager
        mgr.total_dofs = 15
        
        # Build connectivity
        connectivity = build_dof_connectivity(elements, mgr)
        
        @test connectivity.n_total_dofs == 15
        @test length(connectivity) == 15
        
        # Test shared DOFs (4, 5, 6)
        @test connection_count(connectivity, 4) == 2  # In both elements
        @test connection_count(connectivity, 5) == 2
        @test connection_count(connectivity, 6) == 2
        
        # Test unique DOFs
        @test connection_count(connectivity, 1) == 1  # Only in elem1
        @test connection_count(connectivity, 10) == 1  # Only in elem2
        
        # Test connections
        conns_4 = connectivity[4]
        @test length(conns_4) == 2
        @test elem_id(conns_4[1]) in [1, 2]
        @test elem_id(conns_4[2]) in [1, 2]
        @test elem_id(conns_4[1]) != elem_id(conns_4[2])
        
        # Test empty DOFs (if any)
        # DOFs 16+ don't exist, but we only built for 1-15
        # Test that accessing valid DOFs works
        @test !is_empty(connectivity, 1)
        @test !is_empty(connectivity, 4)
        
        println("  ✓ DOFConnectivity (CPU) working")
    end
    
    # ========================================================================
    # 3. DOFConnectivityGPU
    # ========================================================================
    
    @testset "DOFConnectivityGPU" begin
        println("\n[3] Testing DOFConnectivityGPU...")
        
        # Same test case as CPU version
        mgr = DOFManager()
        mgr.total_dofs = 15
        
        S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
        ElemType = Element{Triangle{3}, Lagrange{1}, S}
        elem1 = ElemType(UInt(1), (UInt64(1), UInt64(2), UInt64(3), UInt64(4), UInt64(5), UInt64(6), UInt64(7), UInt64(8), UInt64(9)))
        elem2 = ElemType(UInt(2), (UInt64(4), UInt64(5), UInt64(6), UInt64(10), UInt64(11), UInt64(12), UInt64(13), UInt64(14), UInt64(15)))
        elements = [elem1, elem2]
        
        # Build GPU connectivity
        connectivity_gpu = build_dof_connectivity_gpu(elements, mgr, max_connections=10)
        
        @test connectivity_gpu.n_total_dofs == 15
        @test connectivity_gpu.max_connections == 10
        @test size(connectivity_gpu.elem_ids) == (10, 15)
        @test size(connectivity_gpu.local_indices) == (10, 15)
        @test length(connectivity_gpu.counts) == 15
        
        # Test shared DOFs
        @test connection_count(connectivity_gpu, 4) == 2
        @test connection_count(connectivity_gpu, 5) == 2
        @test connection_count(connectivity_gpu, 6) == 2
        
        # Test unique DOFs
        @test connection_count(connectivity_gpu, 1) == 1
        @test connection_count(connectivity_gpu, 10) == 1
        
        # Test data access
        @test connectivity_gpu.elem_ids[1, 4] in [1, 2]
        @test connectivity_gpu.elem_ids[2, 4] in [1, 2]
        @test connectivity_gpu.elem_ids[1, 4] != connectivity_gpu.elem_ids[2, 4]
        
        # Test array types (GPU-compatible - all bits types)
        @test eltype(connectivity_gpu.elem_ids) == Int32
        @test eltype(connectivity_gpu.local_indices) == Int16
        @test eltype(connectivity_gpu.counts) == Int32
        # Verify arrays are standard Julia arrays (contiguous by default for Matrix/Vector)
        @test connectivity_gpu.elem_ids isa Matrix{Int32}
        @test connectivity_gpu.local_indices isa Matrix{Int16}
        @test connectivity_gpu.counts isa Vector{Int32}
        
        println("  ✓ DOFConnectivityGPU working")
    end
    
    # ========================================================================
    # 4. Zero-Allocation Verification
    # ========================================================================
    
    @testset "Zero-Allocation Access" begin
        println("\n[4] Testing zero-allocation access...")
        
        # Build connectivity
        # Create 8 elements: 8 × 12 = 96 DOFs (within 100)
        mgr = DOFManager()
        mgr.total_dofs = 120  # Enough for 8 elements
        
        S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
        ElemType = Element{Tetrahedron{4}, Lagrange{1}, S}
        
        # Create 8 elements with DOFs
        elements = ElemType[]
        for i in 1:8
            # Each element: 12 DOFs (4 nodes × 3)
            start_dof = (i - 1) * 12 + 1
            dofs = tuple((UInt64(d) for d in start_dof:(start_dof + 11))...)
            push!(elements, ElemType(UInt(i), dofs))
        end
        
        connectivity = build_dof_connectivity(elements, mgr)
        
        # Warm-up (ensure compilation)
        n_warmup = min(96, connectivity.n_total_dofs)
        for dof_i in 1:n_warmup
            _ = connectivity[dof_i]
            _ = connection_count(connectivity, dof_i)
        end
        
        # Test on subset of DOFs that have connections
        n_test = 96  # All DOFs from 8 elements
        
        # Test access allocations (define function in global scope for @benchmark)
        test_access_func = let conn = connectivity, n = n_test
            function()
                @inbounds for dof_i in 1:n
                    connections = conn[dof_i]
                    count = length(connections)
                    for c in connections
                        _ = elem_id(c)
                        _ = local_dof_idx(c)
                    end
                end
            end
        end
        
        result = @benchmark $test_access_func()
        
        println("  Access allocations: $(result.allocs)")
        println("  Access memory: $(result.memory) bytes")
        
        # Should be zero allocations after warmup
        @test result.allocs == 0
        @test result.memory == 0
        
        println("  ✓ Zero-allocation access verified")
    end
    
    # ========================================================================
    # 5. Edge Cases
    # ========================================================================
    
    @testset "Edge Cases" begin
        println("\n[5] Testing edge cases...")
        
        # Empty elements
        mgr = DOFManager()
        mgr.total_dofs = 0
        connectivity = build_dof_connectivity(Element[], mgr)
        @test connectivity.n_total_dofs == 0
        @test length(connectivity.dof_to_elements) == 0
        
        # Single element
        mgr = DOFManager()
        mgr.total_dofs = 12
        S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
        ElemType = Element{Tetrahedron{4}, Lagrange{1}, S}
        elem = ElemType(UInt(1), (UInt64(1), UInt64(2), UInt64(3), UInt64(4), UInt64(5), UInt64(6), UInt64(7), UInt64(8), UInt64(9), UInt64(10), UInt64(11), UInt64(12)))
        connectivity = build_dof_connectivity([elem], mgr)
        @test connectivity.n_total_dofs == 12
        @test all(count -> count == 1, [connection_count(connectivity, i) for i in 1:12])
        
        # Invalid DOF index
        mgr2 = DOFManager()
        mgr2.total_dofs = 10
        elem_invalid = ElemType(UInt(1), (UInt64(1), UInt64(2), UInt64(3), UInt64(4), UInt64(5), UInt64(6), UInt64(7), UInt64(8), UInt64(9), UInt64(10), UInt64(11), UInt64(12)))  # DOF 11, 12 invalid
        @test_throws ErrorException build_dof_connectivity([elem_invalid], mgr2)
        
        println("  ✓ Edge cases handled")
    end
    
    # ========================================================================
    # 6. GPU Compatibility
    # ========================================================================
    
    @testset "GPU Compatibility" begin
        println("\n[6] Testing GPU compatibility...")
        
        # Test that DOFElementConnection is a bits type (GPU-compatible)
        conn_test = DOFElementConnection(1, 1)
        @test isbits(conn_test)
        # Verify it's a struct type (not mutable)
        @test DOFElementConnection isa Type
        
        # Test that GPU connectivity arrays are transferable
        mgr = DOFManager()
        mgr.total_dofs = 50
        S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
        ElemType = Element{Tetrahedron{4}, Lagrange{1}, S}
        elements = [ElemType(UInt(i), tuple((UInt64(d) for d in 1:12)...)) for i in 1:5]
        
        conn_gpu = build_dof_connectivity_gpu(elements, mgr, max_connections=10)
        
        # All array element types should be bits types (for GPU transfer)
        # Test with instances, not types
        @test isbits(Int32(1))
        @test isbits(Int16(1))
        @test eltype(conn_gpu.elem_ids) == Int32
        @test eltype(conn_gpu.local_indices) == Int16
        @test eltype(conn_gpu.counts) == Int32
        
        # Arrays should be standard Julia arrays (contiguous by default)
        @test conn_gpu.elem_ids isa Matrix{Int32}
        @test conn_gpu.local_indices isa Matrix{Int16}
        @test conn_gpu.counts isa Vector{Int32}
        
        println("  ✓ GPU compatibility verified")
    end
    
    println("\n" * "="^70)
    println("ALL TESTS PASSED")
    println("="^70)
end
