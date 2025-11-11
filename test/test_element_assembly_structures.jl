# Tests for traditional element assembly structures
#
# Validates the element-by-element assembly approach

using Test
using Tensors
using LinearAlgebra
using SparseArrays
using Printf

include("../src/element_assembly_structures.jl")

@testset "Element Assembly Structures" begin

    @testset "ElementAssemblyData - Construction" begin
        ndof = 30  # 10 nodes × 3 DOF
        assembly = ElementAssemblyData(ndof, Float64)

        @test assembly.ndof == 30
        @test size(assembly.K_global) == (30, 30)
        @test length(assembly.r_global) == 30
        @test length(assembly.f_int_global) == 30
        @test length(assembly.f_ext_global) == 30

        @test nnz(assembly.K_global) == 0  # Empty initially
        @test all(iszero, assembly.r_global)
        @test all(iszero, assembly.f_int_global)
        @test all(iszero, assembly.f_ext_global)

        println("\n✓ ElementAssemblyData construction")
    end

    @testset "DOF Indexing" begin
        # Single Tet4 element with nodes [1,2,3,4]
        conn = (1, 2, 3, 4)
        gdofs = get_dof_indices(conn, 3)

        expected = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        @test gdofs == expected

        # Element with non-sequential nodes
        conn2 = (5, 7, 12, 15)
        gdofs2 = get_dof_indices(conn2, 3)

        expected2 = [13, 14, 15, 19, 20, 21, 34, 35, 36, 43, 44, 45]
        @test gdofs2 == expected2

        println("✓ DOF indexing")
    end

    @testset "Element Contribution - Single Element" begin
        # Create simple element contribution
        conn = (1, 2, 3, 4)
        gdofs = get_dof_indices(conn, 3)

        contrib = ElementContribution(1, gdofs, Float64)

        @test contrib.element_id == 1
        @test contrib.gdofs == gdofs
        @test size(contrib.K_local) == (12, 12)
        @test length(contrib.f_int_local) == 12
        @test length(contrib.f_ext_local) == 12

        # Fill with test values
        contrib.K_local[1, 1] = 100.0
        contrib.K_local[1, 2] = 50.0
        contrib.f_int_local[1] = 10.0
        contrib.f_ext_local[1] = 5.0

        @test contrib.K_local[1, 1] == 100.0
        @test contrib.f_int_local[1] == 10.0

        println("✓ Element contribution construction")
    end

    @testset "Scatter to Global - Single Element" begin
        ndof = 12  # 4 nodes × 3 DOF
        assembly = ElementAssemblyData(ndof, Float64)

        # Create element contribution
        conn = (1, 2, 3, 4)
        gdofs = get_dof_indices(conn, 3)
        contrib = ElementContribution(1, gdofs, Float64)

        # Fill with identity-like stiffness
        for i in 1:12
            contrib.K_local[i, i] = 1.0
        end
        contrib.f_int_local[1] = 10.0
        contrib.f_ext_local[1] = 5.0

        # Scatter
        scatter_to_global!(assembly, contrib)

        # Check results
        @test assembly.K_global[1, 1] == 1.0
        @test assembly.K_global[6, 6] == 1.0
        @test assembly.f_int_global[1] == 10.0
        @test assembly.f_ext_global[1] == 5.0

        println("✓ Scatter to global (single element)")
    end

    @testset "Scatter to Global - Overlapping Elements" begin
        ndof = 15  # 5 nodes × 3 DOF
        assembly = ElementAssemblyData(ndof, Float64)

        # Element 1: nodes [1,2,3,4]
        conn1 = (1, 2, 3, 4)
        gdofs1 = get_dof_indices(conn1, 3)
        contrib1 = ElementContribution(1, gdofs1, Float64)

        # Element 2: nodes [2,3,4,5] - shares nodes with element 1
        conn2 = (2, 3, 4, 5)
        gdofs2 = get_dof_indices(conn2, 3)
        contrib2 = ElementContribution(2, gdofs2, Float64)

        # Fill element 1
        for i in 1:12
            contrib1.K_local[i, i] = 1.0
        end
        contrib1.f_int_local[4] = 10.0  # DOF 4 (node 2, x-direction)

        # Fill element 2
        for i in 1:12
            contrib2.K_local[i, i] = 2.0
        end
        contrib2.f_int_local[1] = 5.0   # DOF 4 (node 2, x-direction) - same global DOF!

        # Scatter both
        scatter_to_global!(assembly, contrib1)
        scatter_to_global!(assembly, contrib2)

        # Check accumulation
        # Node 2, DOF x (global DOF 4): should have contributions from both elements
        @test assembly.K_global[4, 4] == 1.0 + 2.0  # Diagonal accumulated
        @test assembly.f_int_global[4] == 10.0 + 5.0  # Force accumulated

        # Node 1 (only in element 1)
        @test assembly.K_global[1, 1] == 1.0

        # Node 5 (only in element 2)
        @test assembly.K_global[13, 13] == 2.0

        println("✓ Scatter to global (overlapping elements)")
    end

    @testset "Residual Computation" begin
        ndof = 12
        assembly = ElementAssemblyData(ndof, Float64)

        # Set up simple forces
        assembly.f_int_global[1] = 100.0
        assembly.f_ext_global[1] = 30.0
        assembly.f_int_global[5] = 50.0
        assembly.f_ext_global[5] = 50.0  # Balanced

        compute_residual!(assembly)

        @test assembly.r_global[1] == 70.0   # 100 - 30
        @test assembly.r_global[5] == 0.0    # 50 - 50

        println("✓ Residual computation")
    end

    @testset "Full Assembly Workflow" begin
        ndof = 15  # 5 nodes × 3 DOF
        assembly = ElementAssemblyData(ndof, Float64)

        # Create two elements
        contributions = ElementContribution{Float64}[]

        # Element 1
        conn1 = (1, 2, 3, 4)
        contrib1 = ElementContribution(1, get_dof_indices(conn1, 3), Float64)
        for i in 1:12
            contrib1.K_local[i, i] = 10.0
        end
        contrib1.f_int_local .= 1.0
        contrib1.f_ext_local .= 0.5
        push!(contributions, contrib1)

        # Element 2
        conn2 = (2, 3, 4, 5)
        contrib2 = ElementContribution(2, get_dof_indices(conn2, 3), Float64)
        for i in 1:12
            contrib2.K_local[i, i] = 20.0
        end
        contrib2.f_int_local .= 2.0
        contrib2.f_ext_local .= 1.0
        push!(contributions, contrib2)

        # Assemble all
        assemble_elements!(assembly, contributions)

        # Check results
        # Node 1 (only element 1)
        @test assembly.K_global[1, 1] == 10.0
        @test assembly.f_int_global[1] == 1.0
        @test assembly.r_global[1] == 0.5  # 1.0 - 0.5

        # Node 2 (both elements)
        @test assembly.K_global[4, 4] == 10.0 + 20.0
        @test assembly.f_int_global[4] == 1.0 + 2.0
        @test assembly.r_global[4] == 1.5  # (1+2) - (0.5+1) = 3 - 1.5

        # Node 5 (only element 2)
        @test assembly.K_global[13, 13] == 20.0

        println("✓ Full assembly workflow")
    end

    @testset "Matrix-Vector Product" begin
        ndof = 9  # 3 nodes × 3 DOF
        assembly = ElementAssemblyData(ndof, Float64)

        # Create simple diagonal matrix
        assembly.K_global = spdiagm(0 => ones(9) .* 2.0)

        v = ones(9)
        w = matrix_vector_product(assembly, v)

        @test w ≈ 2.0 .* ones(9)

        # More complex test
        v2 = collect(1.0:9.0)
        w2 = matrix_vector_product(assembly, v2)
        @test w2 ≈ 2.0 .* v2

        println("✓ Matrix-vector product")
    end

    @testset "Dirichlet BC Application" begin
        ndof = 12
        assembly = ElementAssemblyData(ndof, Float64)

        # Create simple stiffness
        assembly.K_global = spdiagm(0 => ones(12) .* 100.0)
        assembly.r_global .= 1.0

        # Fix first node (DOFs 1,2,3) to zero
        fixed_dofs = [1, 2, 3]
        apply_dirichlet_bc!(assembly, fixed_dofs)

        # Check that diagonal increased significantly
        @test assembly.K_global[1, 1] > 1e10
        @test assembly.K_global[2, 2] > 1e10
        @test assembly.K_global[3, 3] > 1e10

        # Unfixed DOFs should be unchanged
        @test assembly.K_global[4, 4] ≈ 100.0

        println("✓ Dirichlet BC application")
    end

    @testset "Symmetry Preservation" begin
        ndof = 12
        assembly = ElementAssemblyData(ndof, Float64)

        # Create symmetric element contribution
        conn = (1, 2, 3, 4)
        contrib = ElementContribution(1, get_dof_indices(conn, 3), Float64)

        # Symmetric matrix
        for i in 1:12, j in 1:12
            contrib.K_local[i, j] = i + j
            contrib.K_local[j, i] = i + j  # Ensure symmetry
        end

        scatter_to_global!(assembly, contrib)

        # Check global matrix is symmetric
        K_full = Matrix(assembly.K_global)
        @test issymmetric(K_full)

        println("✓ Symmetry preservation")
    end

    @testset "Reset Functionality" begin
        ndof = 12
        assembly = ElementAssemblyData(ndof, Float64)

        # Fill with data
        assembly.K_global = spdiagm(0 => ones(12) .* 5.0)
        assembly.f_int_global .= 10.0
        assembly.f_ext_global .= 5.0
        assembly.r_global .= 5.0

        # Reset
        reset!(assembly)

        # Check everything is zero
        @test nnz(assembly.K_global) == 0
        @test all(iszero, assembly.f_int_global)
        @test all(iszero, assembly.f_ext_global)
        @test all(iszero, assembly.r_global)

        println("✓ Reset functionality")
    end

    @testset "Statistics Printing" begin
        ndof = 12
        assembly = ElementAssemblyData(ndof, Float64)

        # Create sparse matrix
        assembly.K_global = spdiagm(0 => ones(12) .* 100.0,
            1 => ones(11) .* 50.0,
            -1 => ones(11) .* 50.0)
        assembly.f_int_global .= 10.0
        assembly.f_ext_global .= 5.0
        compute_residual!(assembly)

        println("\n")
        print_assembly_stats(assembly)

        # Just check it doesn't error
        @test true
    end
end

println("\n" * "="^70)
println("SUMMARY: Traditional Element Assembly Validated ✓")
println("="^70)
println("All element assembly structures working correctly!")
println("Ready for comparison with nodal assembly.")
