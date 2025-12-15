# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Tests for field type unification.

Tests that field types (Displacement{3}, Temperature, etc.) can be used directly
in DOF specifications instead of quantity types (Vec{3}, Float64, etc.).
"""

using Test
using JuliaFEM
using Tensors

@testset "Field Type Unification" begin
    @testset "quantity_type() trait for field types" begin
        # Test quantity_type trait on field types directly
        @test quantity_type(Displacement{3}) == Vec{3}
        @test quantity_type(Displacement{2}) == Vec{2}
        @test quantity_type(Displacement{1}) == Vec{1}
        @test quantity_type(Temperature) == Float64
        @test quantity_type(DisplacementRotation{3}) == Vec{6}
        @test quantity_type(DisplacementRotation{2}) == Vec{4}
    end

    @testset "quantity_type() with field types in tuples" begin
        # Test quantity_type extraction from tuples with field types
        S_u = @DOFSet{u::DOF{Displacement{3}, Vertex}}
        @test quantity_type(fieldtype(S_u, :u)) == Vec{3}
        
        S_T = @DOFSet{T::DOF{Temperature, Vertex}}
        @test quantity_type(fieldtype(S_T, :T)) == Float64
        
        S_ur = @DOFSet{u::DOF{DisplacementRotation{3}, Vertex}}
        @test quantity_type(fieldtype(S_ur, :u)) == Vec{6}
        
        # Multi-field
        S_Tu = @DOFSet{
            T::DOF{Temperature, Vertex},
            u::DOF{Displacement{3}, Vertex}
        }
        @test quantity_type(fieldtype(S_Tu, :T)) == Float64
        @test quantity_type(fieldtype(S_Tu, :u)) == Vec{3}
    end

    @testset "DOF counting with field types" begin
        # Single field: displacement
        S_u = @DOFSet{u::DOF{Displacement{3}, Vertex}}
        @test ndofs(Tetrahedron{4}, S_u) == 12  # 4 nodes × 3 components
        
        # Single field: temperature
        S_T = @DOFSet{T::DOF{Temperature, Vertex}}
        @test ndofs(Tetrahedron{4}, S_T) == 4  # 4 nodes × 1 component
        
        # Single field: displacement-rotation
        S_ur = @DOFSet{u::DOF{DisplacementRotation{3}, Vertex}}
        @test ndofs(Tetrahedron{4}, S_ur) == 24  # 4 nodes × 6 components
        
        # Multi-field: thermo-mechanical
        S_Tu = @DOFSet{
            T::DOF{Temperature, Vertex},
            u::DOF{Displacement{3}, Vertex}
        }
        @test ndofs(Tetrahedron{4}, S_Tu) == 16  # 4 + 12
        
        # 2D displacement
        S_u2d = @DOFSet{u::DOF{Displacement{2}, Vertex}}
        @test ndofs(Triangle{3}, S_u2d) == 6  # 3 nodes × 2 components
    end

    @testset "Error handling for invalid formats" begin
        # Test that non-field types in DOF produce errors
        # Vec{3} is not a field type (should be Displacement{3})
        S_invalid = @DOFSet{u::DOF{Vec{3}, Vertex}}
        
        # This should error because Vec{3} is not a field type (AbstractField)
        @test_throws ErrorException quantity_type(fieldtype(S_invalid, :u))
    end

    @testset "field_type_for_dispatch with field types" begin
        S = @DOFSet{
            T::DOF{Temperature, Vertex},
            u::DOF{Displacement{3}, Vertex}
        }
        
        field_T = field_type_for_dispatch(S, :T)
        @test field_T isa Temperature
        
        field_u = field_type_for_dispatch(S, :u)
        @test field_u isa Displacement{3}
    end

    @testset "DOF manager with field types" begin
        using JuliaFEM: DOFManager, _assign_element_dofs!
        
        mgr = DOFManager()
        S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
        connectivity = (1, 2, 3, 4)  # 4-node tetrahedron
        
        dof_indices = _assign_element_dofs!(mgr, S, Tetrahedron{4}, connectivity)
        
        # Should return NamedTuple with :u field
        @test haskey(dof_indices, :u)
        @test length(dof_indices.u) == 12  # 4 nodes × 3 components
    end

    @testset "Element creation with field types" begin
        S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
        # Element uses flat tuple of UInt64 for dof_indices
        dof_indices = NTuple{12, UInt64}((1:12...,))
        elem = Element{Tetrahedron{4}, Lagrange{1}, S}(UInt(1), dof_indices)
        
        @test elem isa Element{Tetrahedron{4}, Lagrange{1}, S}
        @test length(elem.dof_indices) == 12
    end
end

