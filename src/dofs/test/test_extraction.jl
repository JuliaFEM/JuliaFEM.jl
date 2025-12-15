# Unit Tests for DOF Extraction
# ===============================

using Test
using JuliaFEM
using StaticArrays
using Tensors

@testset "DOF Extraction" begin
    
    @testset "VectorDOF{3} - 4 nodes (Tet4)" begin
        u_global = rand(100)
        dof_indices = (5, 6, 7, 20, 21, 22, 35, 36, 37, 50, 51, 52)
        
        result = extract_element_dofs(VectorDOF{3}, u_global, dof_indices)
        
        @test result isa SVector{4, Vec{3, Float64}}
        @test result[1] == Vec{3}(u_global[5], u_global[6], u_global[7])
        @test result[2] == Vec{3}(u_global[20], u_global[21], u_global[22])
        @test result[3] == Vec{3}(u_global[35], u_global[36], u_global[37])
        @test result[4] == Vec{3}(u_global[50], u_global[51], u_global[52])
    end
    
    @testset "VectorDOF{2} - 3 nodes (Tri3)" begin
        u_global = rand(50)
        dof_indices = (10, 11, 20, 21, 30, 31)
        
        result = extract_element_dofs(VectorDOF{2}, u_global, dof_indices)
        
        @test result isa SVector{3, Vec{2, Float64}}
        @test result[1] == Vec{2}(u_global[10], u_global[11])
        @test result[2] == Vec{2}(u_global[20], u_global[21])
        @test result[3] == Vec{2}(u_global[30], u_global[31])
    end
    
    @testset "ScalarDOF - 4 nodes" begin
        u_global = rand(100)
        dof_indices = (10, 25, 40, 55)
        
        result = extract_element_dofs(ScalarDOF, u_global, dof_indices)
        
        @test result isa SVector{4, Float64}
        @test result[1] == u_global[10]
        @test result[2] == u_global[25]
        @test result[3] == u_global[40]
        @test result[4] == u_global[55]
    end
    
    @testset "Type Stability" begin
        u = rand(100)
        inds = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
        
        # VectorDOF should infer concrete return type
        @inferred extract_element_dofs(VectorDOF{3}, u, inds)
        
        # ScalarDOF should infer concrete return type
        @inferred extract_element_dofs(ScalarDOF, u, (1, 2, 3, 4))
    end
    
    @testset "Zero Allocation" begin
        u = rand(100)
        inds = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
        
        # Warm up
        extract_element_dofs(VectorDOF{3}, u, inds)
        
        # Check GC pressure over many calls
        GC.gc()
        before = GC.gc_num()
        
        for i in 1:1000
            extract_element_dofs(VectorDOF{3}, u, inds)
        end
        
        after = GC.gc_num()
        
        # Should have negligible allocations
        bytes_per_call = (after.allocd - before.allocd) / 1000
        @test bytes_per_call < 10.0  # Less than 10 bytes per call
    end
    
    @testset "Correctness vs Manual" begin
        u = rand(100)
        inds = (5, 6, 7, 20, 21, 22, 35, 36, 37, 50, 51, 52)
        
        # Manual extraction
        manual = SVector(
            Vec{3}((u[inds[1]], u[inds[2]], u[inds[3]])),
            Vec{3}((u[inds[4]], u[inds[5]], u[inds[6]])),
            Vec{3}((u[inds[7]], u[inds[8]], u[inds[9]])),
            Vec{3}((u[inds[10]], u[inds[11]], u[inds[12]]))
        )
        
        # @generated extraction
        generated = extract_element_dofs(VectorDOF{3}, u, inds)
        
        @test generated ≈ manual
        @test generated == manual  # Should be exactly equal
    end
    
end
