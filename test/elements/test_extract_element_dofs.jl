# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using Test
using JuliaFEM
using Tensors

@testset "DOF Extraction: Single-field Scalar" begin
    # Heat equation: 1 scalar per node
    S = @DOFSet{T::DOF{Temperature, Vertex}}
    elem = Element{Triangle{3}, Lagrange{1}, S}(
        UInt(1),
        (UInt64(10), UInt64(20), UInt64(30))  # Flat tuple of DOF indices
    )
    
    # Global solution vector (100 DOFs total)
    u_global = collect(1.0:100.0)
    
    # Extract as flat tuple (NamedTuple wrapper for consistency)
    dofs_flat = extract_element_dofs(elem, u_global)
    @test dofs_flat isa NamedTuple
    @test keys(dofs_flat) == (:T,)
    @test dofs_flat.T == (10.0, 20.0, 30.0)
    
    # Extract structured (same for scalars)
    dofs_struct = extract_element_dofs_structured(elem, u_global)
    @test dofs_struct isa NamedTuple
    @test keys(dofs_struct) == (:T,)
    @test dofs_struct.T == (10.0, 20.0, 30.0)
end

@testset "DOF Extraction: Single-field Vector" begin
    # 2D elasticity: Vec{2} per node, 3 nodes
    S = @DOFSet{u::DOF{Displacement{2}, Vertex}}
    elem = Element{Triangle{3}, Lagrange{1}, S}(
        UInt(1),
        (UInt64(1), UInt64(2), UInt64(3), UInt64(4), UInt64(5), UInt64(6))  # Flat tuple
    )
    
    u_global = collect(1.0:100.0)
    
    # Flat extraction: all scalars (NamedTuple wrapper)
    dofs_flat = extract_element_dofs(elem, u_global)
    @test dofs_flat isa NamedTuple
    @test keys(dofs_flat) == (:u,)
    @test length(dofs_flat.u) == 6
    @test dofs_flat.u == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    
    # Structured extraction: Vec{2} instances (NamedTuple wrapper)
    dofs_struct = extract_element_dofs_structured(elem, u_global)
    @test dofs_struct isa NamedTuple
    @test keys(dofs_struct) == (:u,)
    @test length(dofs_struct.u) == 3  # 3 nodes
    @test dofs_struct.u[1] == Vec(1.0, 2.0)
    @test dofs_struct.u[2] == Vec(3.0, 4.0)
    @test dofs_struct.u[3] == Vec(5.0, 6.0)
end

@testset "DOF Extraction: Single-field Vector 3D" begin
    # 3D elasticity: Vec{3} per node, 4 nodes (tetrahedron)
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    elem = Element{Tetrahedron{4}, Lagrange{1}, S}(
        UInt(1),
        tuple(UInt64.(1:12)...)  # Flat tuple: 12 DOFs
    )
    
    u_global = collect(1.0:100.0)
    
    # Flat extraction: Returns NamedTuple with flat tuple
    dofs_flat = extract_element_dofs(elem, u_global)
    @test dofs_flat isa NamedTuple
    @test keys(dofs_flat) == (:u,)
    @test length(dofs_flat.u) == 12
    @test dofs_flat.u == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0)
    
    # Structured extraction: Returns NamedTuple with Vec{3} instances (one per vertex)
    dofs_struct = extract_element_dofs_structured(elem, u_global)
    @test dofs_struct isa NamedTuple
    @test keys(dofs_struct) == (:u,)
    @test length(dofs_struct.u) == 4  # 4 vertices
    @test dofs_struct.u[1] == Vec(1.0, 2.0, 3.0)
    @test dofs_struct.u[2] == Vec(4.0, 5.0, 6.0)
    @test dofs_struct.u[3] == Vec(7.0, 8.0, 9.0)
    @test dofs_struct.u[4] == Vec(10.0, 11.0, 12.0)
end

@testset "DOF Extraction: Multi-field (Temperature + Displacement)" begin
    # Thermoelasticity: Float64 (T) + Vec{3} (u) at vertices
    S = @DOFSet{T::DOF{Temperature, Vertex}, u::DOF{Displacement{3}, Vertex}}
    elem = Element{Tetrahedron{4}, Lagrange{1}, S}(
        UInt(1),
        (UInt64(1), UInt64(2), UInt64(3), UInt64(4), UInt64(10), UInt64(11), UInt64(12), UInt64(13), UInt64(14), UInt64(15), UInt64(16), UInt64(17), UInt64(18), UInt64(19), UInt64(20), UInt64(21))  # Flat: 4 T + 12 u = 16 DOFs
    )
    
    u_global = collect(1.0:100.0)
    
    # Flat extraction: NamedTuple of tuples
    dofs_flat = extract_element_dofs(elem, u_global)
    @test dofs_flat isa NamedTuple
    @test keys(dofs_flat) == (:T, :u)
    @test dofs_flat.T == (1.0, 2.0, 3.0, 4.0)
    @test dofs_flat.u == (10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0)
    
    # Structured extraction: NamedTuple with Vec{3} for displacement
    dofs_struct = extract_element_dofs_structured(elem, u_global)
    @test dofs_struct isa NamedTuple
    @test keys(dofs_struct) == (:T, :u)
    @test dofs_struct.T == (1.0, 2.0, 3.0, 4.0)
    @test length(dofs_struct.u) == 4  # 4 nodes
    @test dofs_struct.u[1] == Vec(10.0, 11.0, 12.0)
    @test dofs_struct.u[2] == Vec(13.0, 14.0, 15.0)
    @test dofs_struct.u[3] == Vec(16.0, 17.0, 18.0)
    @test dofs_struct.u[4] == Vec(19.0, 20.0, 21.0)
end

@testset "DOF Extraction: Non-contiguous indices" begin
    # Test with scattered global DOF indices
    S = @DOFSet{u::DOF{Displacement{2}, Vertex}}
    elem = Element{Triangle{3}, Lagrange{1}, S}(
        UInt(1),
        (UInt64(5), UInt64(12), UInt64(23), UInt64(34), UInt64(45), UInt64(56))  # Non-sequential indices
    )
    
    u_global = collect(1.0:100.0)
    
    # Flat extraction
    dofs_flat = extract_element_dofs(elem, u_global)
    @test dofs_flat isa NamedTuple
    @test keys(dofs_flat) == (:u,)
    @test dofs_flat.u == (5.0, 12.0, 23.0, 34.0, 45.0, 56.0)
    
    # Structured extraction
    dofs_struct = extract_element_dofs_structured(elem, u_global)
    @test dofs_struct isa NamedTuple
    @test keys(dofs_struct) == (:u,)
    @test dofs_struct.u[1] == Vec(5.0, 12.0)
    @test dofs_struct.u[2] == Vec(23.0, 34.0)
    @test dofs_struct.u[3] == Vec(45.0, 56.0)
end

@testset "Zero Allocation: All extraction functions" begin
    # Single-field scalar
    S_scalar = @DOFSet{T::DOF{Temperature, Vertex}}
    elem_scalar = Element{Triangle{3}, Lagrange{1}, S_scalar}(
        UInt(1),
        (UInt64(10), UInt64(20), UInt64(30))
    )
    u_global = collect(1.0:100.0)
    
    # Warm up
    extract_element_dofs(elem_scalar, u_global)
    extract_element_dofs_structured(elem_scalar, u_global)
    
    # Check zero allocation
    alloc_flat = @allocated extract_element_dofs(elem_scalar, u_global)
    alloc_struct = @allocated extract_element_dofs_structured(elem_scalar, u_global)
    @test alloc_flat == 0
    @test alloc_struct == 0
    
    # Single-field vector 2D
    S_vec2d = @DOFSet{u::DOF{Displacement{2}, Vertex}}
    elem_vec2d = Element{Triangle{3}, Lagrange{1}, S_vec2d}(
        UInt(1),
        tuple(UInt64.(1:6)...)
    )
    
    # Warm up
    extract_element_dofs(elem_vec2d, u_global)
    extract_element_dofs_structured(elem_vec2d, u_global)
    
    # Check zero allocation
    alloc_flat = @allocated extract_element_dofs(elem_vec2d, u_global)
    alloc_struct = @allocated extract_element_dofs_structured(elem_vec2d, u_global)
    @test alloc_flat == 0
    @test alloc_struct == 0
    
    # Single-field vector 3D
    S_vec3d = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    elem_vec3d = Element{Tetrahedron{4}, Lagrange{1}, S_vec3d}(
        UInt(1),
        tuple(UInt64.(1:12)...)
    )
    
    # Warm up
    extract_element_dofs(elem_vec3d, u_global)
    extract_element_dofs_structured(elem_vec3d, u_global)
    
    # Check zero allocation
    alloc_flat = @allocated extract_element_dofs(elem_vec3d, u_global)
    alloc_struct = @allocated extract_element_dofs_structured(elem_vec3d, u_global)
    @test alloc_flat == 0
    @test alloc_struct == 0
    
    # Multi-field (Temperature + Displacement)
    S_multi = @DOFSet{T::DOF{Temperature,Vertex}, u::DOF{Displacement{3},Vertex}}
    elem_multi = Element{Tetrahedron{4}, Lagrange{1}, S_multi}(
        UInt(1),
        tuple(UInt64.([1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])...)
    )
    
    # Warm up
    extract_element_dofs(elem_multi, u_global)
    extract_element_dofs_structured(elem_multi, u_global)
    
    # Check zero allocation
    alloc_flat = @allocated extract_element_dofs(elem_multi, u_global)
    alloc_struct = @allocated extract_element_dofs_structured(elem_multi, u_global)
    @test alloc_flat == 0
    @test alloc_struct == 0
end

println("✓ All DOF extraction tests passed!")
println("  - Single-field scalar extraction works")
println("  - Single-field vector extraction works")
println("  - Multi-field extraction works")
println("  - Non-contiguous indices work")
println("  - Flat and structured variants both work")
println("  - Zero allocations verified for all cases")
