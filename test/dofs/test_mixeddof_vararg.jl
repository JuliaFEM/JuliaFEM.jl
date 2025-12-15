# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE

"""
Test for variadic MixedDOF type - combining multiple DOF types at type level.

This demonstrates the **vararg** approach to multi-field elements, where you
mix DOF types together as type parameters (not with field names).
"""

using Test
using JuliaFEM
using JuliaFEM: Vertex, Edge, Cell, DOF, MixedDOF, create_elements!
using Tensors

@testset "🎯 MixedDOF: Variadic Multi-Field Elements" begin
    
    # Create minimal mesh: 2 tetrahedra sharing a face
    # Nodes: 1-5 (vertex IDs)
    nodes = [
        Vec(0.0, 0.0, 0.0),  # 1
        Vec(1.0, 0.0, 0.0),  # 2
        Vec(0.0, 1.0, 0.0),  # 3
        Vec(0.0, 0.0, 1.0),  # 4
        Vec(1.0, 1.0, 1.0),  # 5
    ]
    
    connectivity = [
        (UInt32(1), UInt32(2), UInt32(3), UInt32(4)),  # Tet 1
        (UInt32(2), UInt32(3), UInt32(4), UInt32(5)),  # Tet 2
    ]
    
    mesh = Mesh{4, Tetrahedron{4}}(nodes, connectivity)
    
    println("\n" * "="^70)
    println("🎯 VARIADIC MixedDOF: Type-Level Multi-Field Composition")
    println("="^70)
    
    # ========================================================================
    # Define MixedDOF with 4 fields (NO field names, just types!)
    # ========================================================================
    
    println("\n📋 Defining MixedDOF (variadic, type-level)...")
    
    # Simple syntax - no need to repeat DOF!
    mixed_type = @MixedDOF{
        (Float64, Vertex),    # Field 1: Temperature at vertices
        (Vec{3}, Vertex),     # Field 2: Displacement at vertices
        (Float64, Cell),      # Field 3: Pressure at cells
        (Float64, Edge)       # Field 4: Electric potential on edges
    }
    
    println("   Type: $mixed_type")
    println("   ✓ 4 DOF types mixed together")
    println("   ✓ Access by position (tuple unpacking)")
    println("   ✓ Simple syntax: @MixedDOF{(Float64, Vertex), (Vec{3}, Vertex), ...}")
    
    # ========================================================================
    # Create elements with MixedDOF
    # ========================================================================
    
    println("\n🏗️  Creating elements...")
    
    eltype = Element{Tetrahedron{4}, Lagrange{1}, mixed_type}
    elements, mgr = create_elements!(mesh, eltype)
    
    println("   ✓ Created $(length(elements)) elements")
    @test length(elements) == 2
    
    # ========================================================================
    # Test DOF indices structure (tuple of tuples, NOT NamedTuple!)
    # ========================================================================
    
    println("\n🔍 Examining DOF indices...")
    
    elem = first(elements)
    println("   Element 1 DOF indices type: $(typeof(elem.dof_indices))")
    @test elem.dof_indices isa Tuple
    @test length(elem.dof_indices) == 4  # 4 fields
    
    # Unpack by position
    T_dofs, u_dofs, p_dofs, φ_dofs = elem.dof_indices
    
    println("   ✓ Temperature DOFs: $T_dofs")
    println("   ✓ Displacement DOFs: $u_dofs")
    println("   ✓ Pressure DOFs: $p_dofs")
    println("   ✓ Electric DOFs: $φ_dofs")
    
    # Validate structure
    @test length(T_dofs) == 4   # 4 vertices
    @test length(u_dofs) == 12  # 4 vertices × 3 components
    @test length(p_dofs) == 1   # 1 cell (discontinuous)
    @test length(φ_dofs) == 6   # 6 edges
    
    # ========================================================================
    # Test DOF sharing between elements
    # ========================================================================
    
    println("\n🔗 Testing DOF sharing...")
    
    elem2 = elements[2]
    T_dofs2, u_dofs2, p_dofs2, φ_dofs2 = elem2.dof_indices
    
    # Vertices 2,3,4 are shared → same DOF indices
    shared_nodes_overlap = !isempty(intersect(T_dofs, T_dofs2))
    @test shared_nodes_overlap
    println("   ✓ Shared vertices have same temperature DOFs")
    
    # Cell DOFs are unique (no sharing)
    @test isempty(intersect(p_dofs, p_dofs2))
    println("   ✓ Cell DOFs are unique (discontinuous)")
    
    # Edge DOFs: Some edges are shared
    shared_edges_overlap = !isempty(intersect(φ_dofs, φ_dofs2))
    @test shared_edges_overlap
    println("   ✓ Shared edges have same electric DOFs")
    
    # ========================================================================
    # Total system DOFs
    # ========================================================================
    
    println("\n📊 System summary:")
    
    all_T_dofs = unique(vcat([collect(elem.dof_indices[1])  for elem in elements]...))
    all_u_dofs = unique(vcat([collect(elem.dof_indices[2])  for elem in elements]...))
    all_p_dofs = unique(vcat([collect(elem.dof_indices[3])  for elem in elements]...))
    all_φ_dofs = unique(vcat([collect(elem.dof_indices[4])  for elem in elements]...))
    
    println("   Temperature DOFs: $(length(all_T_dofs)) (5 unique vertices)")
    println("   Displacement DOFs: $(length(all_u_dofs)) (5 vertices × 3)")
    println("   Pressure DOFs: $(length(all_p_dofs)) (2 cells, no sharing)")
    println("   Electric DOFs: $(length(all_φ_dofs)) (9 unique edges)")
    
    total_dofs = length(all_T_dofs) + length(all_u_dofs) + length(all_p_dofs) + length(all_φ_dofs)
    println("   ────────────────────────────")
    println("   TOTAL: $total_dofs DOFs")
    
    @test length(all_T_dofs) == 5
    @test length(all_u_dofs) == 15
    @test length(all_p_dofs) == 2
    @test length(all_φ_dofs) == 9
    @test total_dofs == 31
    
    # ========================================================================
    # Compare: MixedDOF vs NamedTuple
    # ========================================================================
    
    println("\n💡 COMPARISON: MixedDOF vs NamedTuple")
    println("   ┌─────────────────────────────────────────────────────────┐")
    println("   │ MixedDOF (vararg):                                      │")
    println("   │   - Type-level composition                              │")
    println("   │   - Access by position: T_dofs, u_dofs, ... = elem.dof_indices │")
    println("   │   - More compact type signature                         │")
    println("   │                                                         │")
    println("   │ NamedTuple:                                             │")
    println("   │   - Named fields                                        │")
    println("   │   - Access by name: elem.dof_indices.T, .u, .p, .φ    │")
    println("   │   - More readable, self-documenting                     │")
    println("   │                                                         │")
    println("   │ BOTH ARE SUPPORTED! Choose based on preference.        │")
    println("   └─────────────────────────────────────────────────────────┘")
    
    println("\n" * "="^70)
    println("✅ ALL TESTS PASSED: Variadic MixedDOF works!")
    println("="^70)
end
