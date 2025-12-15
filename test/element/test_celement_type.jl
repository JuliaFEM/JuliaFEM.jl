# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE

"""
Test CElement Type Properties
"""

@testset "CElement Type Properties" begin
    @testset "Construction and Validation" begin
        # Valid construction: Tet4 with VectorDOF{3}
        # 4 nodes × 3 DOFs/node = 12 DOFs
        elem = CElement{Tetrahedron{4}, Lagrange{1}, VectorDOF{3}}(
            1,
            (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
        )
        
        @test elem isa CElement
        @test element_id(elem) == 1
        @test length(element_dofs(elem)) == 12
        @test n_element_dofs(elem) == 12
    end
    
    @testset "DOF Count Validation" begin
        # Should error: wrong number of DOF indices
        # Tet4 with VectorDOF{3} needs 12 DOFs, not 9
        @test_throws ErrorException CElement{Tetrahedron{4}, Lagrange{1}, VectorDOF{3}}(
            1,
            (1, 2, 3, 4, 5, 6, 7, 8, 9)  # Only 9 DOFs!
        )
    end
    
    @testset "Type Queries" begin
        elem = CElement{Triangle{3}, Lagrange{1}, ScalarDOF}(
            1,
            (1, 2, 3)
        )
        
        @test topology_type(elem) == Triangle{3}
        @test basis_type(elem) == Lagrange{1}
        @test dof_type(elem) == ScalarDOF
        
        # Also works on type itself
        @test topology_type(typeof(elem)) == Triangle{3}
        @test basis_type(typeof(elem)) == Lagrange{1}
        @test dof_type(typeof(elem)) == ScalarDOF
    end
    
    @testset "Different DOF Types" begin
        # Scalar DOF (heat transfer)
        elem_scalar = CElement{Triangle{3}, Lagrange{1}, ScalarDOF}(
            1,
            (5, 12, 23)  # 3 nodes × 1 DOF = 3 DOFs
        )
        @test n_element_dofs(elem_scalar) == 3
        
        # Vector DOF 2D (plane stress)
        elem_vec2d = CElement{Triangle{3}, Lagrange{1}, VectorDOF{2}}(
            2,
            (1, 2, 3, 4, 5, 6)  # 3 nodes × 2 DOFs = 6 DOFs
        )
        @test n_element_dofs(elem_vec2d) == 6
        
        # Vector DOF 3D (solid mechanics)
        elem_vec3d = CElement{Tetrahedron{4}, Lagrange{1}, VectorDOF{3}}(
            3,
            (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)  # 4 nodes × 3 DOFs = 12 DOFs
        )
        @test n_element_dofs(elem_vec3d) == 12
    end
    
    @testset "Display" begin
        elem = CElement{Tetrahedron{4}, Lagrange{1}, VectorDOF{3}}(
            123,
            (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
        )
        
        str = sprint(show, elem)
        @test occursin("CElement", str)
        # Note: Topology type displays as "Tet4" not "Tetrahedron{4}"
        @test occursin("Tet4", str)
        @test occursin("Lagrange{1}", str)
        @test occursin("VectorDOF{3}", str)
        @test occursin("id=123", str)
        @test occursin("ndofs=12", str)
    end
end
