# Test that nbasis functions are zero-cost (compile to constants)

using Test
using JuliaFEM

@testset "nbasis API" begin
    # Test that nbasis is exported and works
    @test nbasis(Triangle{3}(), Lagrange{1}()) == 3
    @test nbasis(Triangle{6}(), Lagrange{2}()) == 6
    @test nbasis(Tetrahedron{4}(), Lagrange{1}()) == 4
    @test nbasis(Tetrahedron{10}(), Lagrange{2}()) == 10
    @test nbasis(Hexahedron{8}(), Lagrange{1}()) == 8
    @test nbasis(Hexahedron{27}(), Lagrange{2}()) == 27
    
    # Serendipity families
    @test nbasis(Quadrilateral{8}(), Serendipity{2}()) == 8
    @test nbasis(Hexahedron{20}(), Serendipity{2}()) == 20
end

@testset "nbasis is zero-cost" begin
    # Verify that nbasis compiles to a constant
    # We use @allocated to check - should be 0 bytes
    
    @test (@allocated nbasis(Triangle{3}(), Lagrange{1}())) == 0
    @test (@allocated nbasis(Tetrahedron{10}(), Lagrange{2}())) == 0
    @test (@allocated nbasis(Quadrilateral{8}(), Serendipity{2}())) == 0
    
    # Also verify with @code_llvm that it returns a constant
    # (This would be done manually during development)
    # @code_llvm nbasis(Triangle{3}(), Lagrange{1}())  # Should show: ret i64 3
end

@testset "validate_dof_consistency" begin
    # Test that validation works correctly
    
    # Valid cases (should not throw)
    @test validate_dof_consistency(ScalarDOF(), Triangle{3}, Lagrange{1}) === nothing
    
    # Invalid cases: wrong number of DOFs (should throw ArgumentError)
    @test_throws ArgumentError validate_dof_consistency(
        VectorDOF{2}(),  # 2 components per node × 3 nodes = 6 DOFs
        Triangle{3},     # 3 nodes
        Lagrange{1}      # 3 basis functions
        # Mismatch: 6 DOFs ≠ 3 basis functions
    )
    
    @test_throws ArgumentError validate_dof_consistency(
        VectorDOF{3}(),    # 3 components per node × 10 nodes = 30 DOFs
        Tetrahedron{10},   # 10 nodes
        Lagrange{2}        # 10 basis functions
        # Mismatch: 30 DOFs ≠ 10 basis functions
    )
    
    # Test error message content for one case
    ex = try
        validate_dof_consistency(VectorDOF{2}(), Triangle{3}, Lagrange{1})
        nothing
    catch e
        e
    end
    @test ex isa ArgumentError
    @test occursin("3 basis functions", ex.msg)
    @test occursin("6 DOFs", ex.msg)
    @test occursin("Ciarlet triplet", ex.msg)
end

@testset "nbasis consistency with get_basis_functions" begin
    # Verify that nbasis returns the same count as actual basis functions
    
    using StaticArrays: SVector
    using Tensors: Vec
    
    test_cases = [
        (Lagrange{1}, Triangle{3}, Vec(0.25, 0.25)),
        (Lagrange{2}, Triangle{6}, Vec(1/3, 1/3)),
        (Lagrange{1}, Tetrahedron{4}, Vec(0.25, 0.25, 0.25)),
        (Lagrange{1}, Hexahedron{8}, Vec(0.0, 0.0, 0.0)),
    ]
    
    for (basis_type, topo_type, xi) in test_cases
        topo = topo_type()
        basis = basis_type()
        
        # Get actual basis function count
        N_actual = get_basis_functions(topo, basis, xi)
        n_actual = length(N_actual)
        
        # Get count from nbasis
        n_expected = nbasis(topo, basis)
        
        @test n_actual == n_expected
    end
end
