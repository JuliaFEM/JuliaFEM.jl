# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Simple unit tests for DOF-based assembler to debug correctness issues.

These tests are designed to be minimal and focused on specific aspects:
1. Single element correctness
2. Two-element correctness (shared node)
3. DOF mapping verification
4. Entry-by-entry comparison
"""

using Test
using JuliaFEM
using JuliaFEM: DOFBasedCOOAssembler, DOFBasedCOOCache
using JuliaFEM: COOAssembler, COOCache, create_cache as create_coo_cache
using JuliaFEM: DOFManager, create_elements!, @DOFSet, DOF, Displacement, Vertex
using LinearAlgebra
using SparseArrays

@testset "DOF-Based Assembler - Simple Debug Tests" begin
    println("\n" * "="^70)
    println("DOF-BASED ASSEMBLER - SIMPLE DEBUG TESTS")
    println("="^70)
    
    # ========================================================================
    # Test 1: Single Tetrahedron - Full Matrix Comparison
    # ========================================================================
    
    @testset "Test 1: Single Tetrahedron" begin
        println("\n[Test 1] Single tetrahedron element...")
        
        # Create single tetrahedron
        nodes = Vec{3,Float64}[
            Vec{3}((0.0, 0.0, 0.0)),
            Vec{3}((1.0, 0.0, 0.0)),
            Vec{3}((0.5, 1.0, 0.0)),
            Vec{3}((0.5, 0.5, 1.0)),
        ]
        connectivity = [(UInt32(1), UInt32(2), UInt32(3), UInt32(4))]
        mesh = Mesh{Tetrahedron{4}}(nodes, connectivity)
        
        # Material and kernel
        material = LinearElastic(E=210e9, ν=0.3)
        kernel = ContinuumKernel(
            ContinuumFormulation{FullThreeD}(),
            material,
            Displacement{3}()
        )
        
        # Create elements
        S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
        elements, dof_mgr = create_elements!(mesh, Element{Tetrahedron{4}, Lagrange{1}, S})
        
        # DOF-based assembly
        assembler_dof = DOFBasedCOOAssembler()
        cache_dof = DOFBasedCOOCache(elements, dof_mgr, mesh, kernel)
        assemble!(cache_dof, assembler_dof, kernel, mesh)
        K_dof, f_dof = extract_system(cache_dof)
        
        # Element-based assembly
        assembler_elem = COOAssembler()
        cache_elem = create_coo_cache(assembler_elem, mesh, kernel)
        assemble!(cache_elem, assembler_elem, kernel, mesh)
        K_elem, f_elem = extract_system(cache_elem)
        
        # Compare
        K_dof_dense = Matrix(K_dof)
        K_elem_dense = Matrix(K_elem)
        diff = K_dof_dense - K_elem_dense
        max_diff = maximum(abs.(diff))
        max_val = maximum(abs.(K_elem_dense))
        rel_diff = max_diff / (max_val + 1e-10)
        
        println("  Matrix size: $(size(K_dof))")
        println("  Max absolute difference: $max_diff")
        println("  Max matrix value: $max_val")
        println("  Max relative difference: $rel_diff")
        
        # Print first few entries if they differ
        if max_diff > 1e-6
            println("  First differing entries:")
            count = 0
            for i in 1:min(12, size(K_dof, 1))
                for j in 1:min(12, size(K_dof, 2))
                    if abs(diff[i, j]) > 1e-6
                        println("    K[$i,$j]: elem=$(K_elem_dense[i,j]), dof=$(K_dof_dense[i,j]), diff=$(diff[i,j])")
                        count += 1
                        if count >= 10
                            break
                        end
                    end
                end
                count >= 10 && break
            end
        end
        
        @test size(K_dof) == size(K_elem)
        @test nnz(K_dof) == nnz(K_elem)
        @test max_diff < 1e-6 || rel_diff < 1e-9
        @test norm(f_dof - f_elem) < 1e-10
    end
    
    # ========================================================================
    # Test 2: Two Tetrahedra Sharing a Face
    # ========================================================================
    
    @testset "Test 2: Two Tetrahedra (Shared Face)" begin
        println("\n[Test 2] Two tetrahedra sharing a face...")
        
        # Two tetrahedra sharing face (1,2,3)
        nodes = Vec{3,Float64}[
            Vec{3}((0.0, 0.0, 0.0)),  # 1
            Vec{3}((1.0, 0.0, 0.0)),  # 2
            Vec{3}((0.5, 1.0, 0.0)),  # 3
            Vec{3}((0.5, 0.5, 1.0)),  # 4
            Vec{3}((0.5, 0.5, -1.0)), # 5
        ]
        connectivity = [
            (UInt32(1), UInt32(2), UInt32(3), UInt32(4)),  # Element 1
            (UInt32(1), UInt32(2), UInt32(3), UInt32(5)),  # Element 2
        ]
        mesh = Mesh{Tetrahedron{4}}(nodes, connectivity)
        
        # Material and kernel
        material = LinearElastic(E=210e9, ν=0.3)
        kernel = ContinuumKernel(
            ContinuumFormulation{FullThreeD}(),
            material,
            Displacement{3}()
        )
        
        # Create elements
        S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
        elements, dof_mgr = create_elements!(mesh, Element{Tetrahedron{4}, Lagrange{1}, S})
        
        println("  Elements: $(length(elements))")
        println("  Total DOFs: $(dof_mgr.total_dofs)")
        println("  Element 1 DOFs: $(element_dofs(elements[1]))")
        println("  Element 2 DOFs: $(element_dofs(elements[2]))")
        
        # DOF-based assembly
        assembler_dof = DOFBasedCOOAssembler()
        cache_dof = DOFBasedCOOCache(elements, dof_mgr, mesh, kernel)
        assemble!(cache_dof, assembler_dof, kernel, mesh)
        K_dof, f_dof = extract_system(cache_dof)
        
        # Element-based assembly
        assembler_elem = COOAssembler()
        cache_elem = create_coo_cache(assembler_elem, mesh, kernel)
        assemble!(cache_elem, assembler_elem, kernel, mesh)
        K_elem, f_elem = extract_system(cache_elem)
        
        # Compare
        K_dof_dense = Matrix(K_dof)
        K_elem_dense = Matrix(K_elem)
        diff = K_dof_dense - K_elem_dense
        max_diff = maximum(abs.(diff))
        max_val = maximum(abs.(K_elem_dense))
        rel_diff = max_diff / (max_val + 1e-10)
        
        println("  Matrix size: $(size(K_dof))")
        println("  Max absolute difference: $max_diff")
        println("  Max relative difference: $rel_diff")
        
        # Check shared node entries (nodes 1, 2, 3 are shared)
        # These should have contributions from both elements
        shared_node_dofs = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # DOFs for nodes 1, 2, 3
        println("  Checking shared node entries (DOFs 1-9):")
        max_shared_diff = 0.0
        for i in shared_node_dofs, j in shared_node_dofs
            d = abs(diff[i, j])
            if d > max_shared_diff
                max_shared_diff = d
            end
            if d > 1e-6
                println("    K[$i,$j]: elem=$(K_elem_dense[i,j]), dof=$(K_dof_dense[i,j]), diff=$d")
            end
        end
        println("  Max difference in shared node block: $max_shared_diff")
        
        @test size(K_dof) == size(K_elem)
        @test nnz(K_dof) == nnz(K_elem)
        @test max_diff < 1e-6 || rel_diff < 1e-9
    end
    
    # ========================================================================
    # Test 3: Single Hex8 Element
    # ========================================================================
    
    @testset "Test 3: Single Hex8 Element" begin
        println("\n[Test 3] Single Hex8 element...")
        
        # Create single Hex8 element
        nodes = Vec{3,Float64}[
            Vec{3}((0.0, 0.0, 0.0)),  # 1
            Vec{3}((1.0, 0.0, 0.0)),  # 2
            Vec{3}((1.0, 1.0, 0.0)),  # 3
            Vec{3}((0.0, 1.0, 0.0)),  # 4
            Vec{3}((0.0, 0.0, 1.0)),  # 5
            Vec{3}((1.0, 0.0, 1.0)),  # 6
            Vec{3}((1.0, 1.0, 1.0)),  # 7
            Vec{3}((0.0, 1.0, 1.0)),  # 8
        ]
        connectivity = [(UInt32(1), UInt32(2), UInt32(3), UInt32(4),
                         UInt32(5), UInt32(6), UInt32(7), UInt32(8))]
        mesh = Mesh{8,Hexahedron{8}}(nodes, connectivity)
        
        # Material and kernel
        material = LinearElastic(E=210e9, ν=0.3)
        kernel = ContinuumKernel(
            ContinuumFormulation{FullThreeD}(),
            material,
            Displacement{3}()
        )
        
        # Create elements
        S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
        elements, dof_mgr = create_elements!(mesh, Element{Hexahedron{8}, Lagrange{1}, S})
        
        println("  Elements: $(length(elements))")
        println("  Total DOFs: $(dof_mgr.total_dofs)")
        println("  Element DOFs: $(element_dofs(elements[1]))")
        
        # DOF-based assembly
        assembler_dof = DOFBasedCOOAssembler()
        cache_dof = DOFBasedCOOCache(elements, dof_mgr, mesh, kernel)
        assemble!(cache_dof, assembler_dof, kernel, mesh)
        K_dof, f_dof = extract_system(cache_dof)
        
        # Element-based assembly
        assembler_elem = COOAssembler()
        cache_elem = create_coo_cache(assembler_elem, mesh, kernel)
        assemble!(cache_elem, assembler_elem, kernel, mesh)
        K_elem, f_elem = extract_system(cache_elem)
        
        # Compare
        K_dof_dense = Matrix(K_dof)
        K_elem_dense = Matrix(K_elem)
        diff = K_dof_dense - K_elem_dense
        max_diff = maximum(abs.(diff))
        max_val = maximum(abs.(K_elem_dense))
        rel_diff = max_diff / (max_val + 1e-10)
        
        println("  Matrix size: $(size(K_dof))")
        println("  Max absolute difference: $max_diff")
        println("  Max relative difference: $rel_diff")
        
        @test size(K_dof) == size(K_elem)
        @test nnz(K_dof) == nnz(K_elem)
        @test max_diff < 1e-6 || rel_diff < 1e-9
    end
    
    # ========================================================================
    # Test 4: Two Hex8 Elements (2×1×1 mesh)
    # ========================================================================
    
    @testset "Test 4: Two Hex8 Elements (2×1×1)" begin
        println("\n[Test 4] Two Hex8 elements (2×1×1 mesh)...")
        
        # Create 2×1×1 mesh (2 elements along X)
        nodes = Vec{3,Float64}[]
        for iz in 0:1, iy in 0:1, ix in 0:2
            push!(nodes, Vec{3}((Float64(ix), Float64(iy), Float64(iz))))
        end
        
        connectivity = NTuple{8,UInt32}[]
        for iz in 0:0, iy in 0:0, ix in 0:1
            # Bottom face
            n1 = ix + iy * 3 + iz * 3 * 2 + 1
            n2 = (ix + 1) + iy * 3 + iz * 3 * 2 + 1
            n3 = (ix + 1) + (iy + 1) * 3 + iz * 3 * 2 + 1
            n4 = ix + (iy + 1) * 3 + iz * 3 * 2 + 1
            # Top face
            n5 = ix + iy * 3 + (iz + 1) * 3 * 2 + 1
            n6 = (ix + 1) + iy * 3 + (iz + 1) * 3 * 2 + 1
            n7 = (ix + 1) + (iy + 1) * 3 + (iz + 1) * 3 * 2 + 1
            n8 = ix + (iy + 1) * 3 + (iz + 1) * 3 * 2 + 1
            push!(connectivity, (n1, n2, n3, n4, n5, n6, n7, n8))
        end
        
        element_sets = Dict{Symbol,Set{UInt32}}(:all => Set(UInt32(1):UInt32(2)))
        mesh = Mesh{8,Hexahedron{8}}(nodes, connectivity, element_sets)
        
        # Material and kernel
        material = LinearElastic(E=210e9, ν=0.3)
        kernel = ContinuumKernel(
            ContinuumFormulation{FullThreeD}(),
            material,
            Displacement{3}()
        )
        
        # Create elements
        S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
        elements, dof_mgr = create_elements!(mesh, Element{Hexahedron{8}, Lagrange{1}, S})
        
        println("  Nodes: $(length(nodes))")
        println("  Elements: $(length(elements))")
        println("  Total DOFs: $(dof_mgr.total_dofs)")
        
        # DOF-based assembly
        assembler_dof = DOFBasedCOOAssembler()
        cache_dof = DOFBasedCOOCache(elements, dof_mgr, mesh, kernel)
        assemble!(cache_dof, assembler_dof, kernel, mesh)
        K_dof, f_dof = extract_system(cache_dof)
        
        # Element-based assembly
        assembler_elem = COOAssembler()
        cache_elem = create_coo_cache(assembler_elem, mesh, kernel)
        assemble!(cache_elem, assembler_elem, kernel, mesh)
        K_elem, f_elem = extract_system(cache_elem)
        
        # Compare
        K_dof_dense = Matrix(K_dof)
        K_elem_dense = Matrix(K_elem)
        diff = K_dof_dense - K_elem_dense
        max_diff = maximum(abs.(diff))
        max_val = maximum(abs.(K_elem_dense))
        rel_diff = max_diff / (max_val + 1e-10)
        
        println("  Matrix size: $(size(K_dof))")
        println("  Max absolute difference: $max_diff")
        println("  Max relative difference: $rel_diff")
        
        # Check shared face entries (nodes 2, 3, 6, 7 are shared between elements)
        # Node 2: DOFs 4,5,6; Node 3: DOFs 7,8,9; Node 6: DOFs 16,17,18; Node 7: DOFs 19,20,21
        shared_dofs = [4, 5, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21]
        println("  Checking shared face entries:")
        max_shared_diff = 0.0
        count = 0
        for i in shared_dofs, j in shared_dofs
            d = abs(diff[i, j])
            if d > max_shared_diff
                max_shared_diff = d
            end
            if d > 1e-6 && count < 5
                println("    K[$i,$j]: elem=$(K_elem_dense[i,j]), dof=$(K_dof_dense[i,j]), diff=$d")
                count += 1
            end
        end
        println("  Max difference in shared face block: $max_shared_diff")
        
        @test size(K_dof) == size(K_elem)
        @test nnz(K_dof) == nnz(K_elem)
        @test max_diff < 1e-6 || rel_diff < 1e-9
    end
    
    println("\n" * "="^70)
    println("ALL SIMPLE DEBUG TESTS COMPLETE")
    println("="^70)
end

